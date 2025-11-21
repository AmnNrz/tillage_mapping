# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + id="whtD--m_FObX"
# Initialize GEE python API
import ee
import geemap
import numpy as np
import geopandas as gpd
import pandas as pd
import time
import os
import re

# ======== DRIVE-ONLY EXPORT → WAIT → DOWNLOAD → DELETE (NO GCS) ========
import io
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.auth import default
import sys

# +
# =============================================================================
# Google Earth Engine and Google Drive Authentication Setup
# =============================================================================
# Path to your service account key file for Drive API authentication
SERVICE_ACCOUNT_FILE = "/path/to/your/service_account_key.json"

# Set the path to your service account JSON file for Google Cloud authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE

# Obtain default credentials with required scopes for both Earth Engine and Drive API access
# Create a credentials object with the necessary scopes
credentials, _ = default(
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform",  # Required for Earth Engine
        "https://www.googleapis.com/auth/drive",  # Required for Google Drive operations
    ]
)

# Initialize Earth Engine API with your Google Cloud project
ee.Initialize(credentials=credentials, project="your-project-id")

# =============================================================================
# Google Drive Export Configuration
# =============================================================================
# Required OAuth2 scopes for Google Drive operations
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Drive folder configuration
# Set to None to export to root directory, or specify a folder name
FOLDER = None  # Example: f"EE_Exports_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Unique timestamp-based tag for file naming to prevent conflicts
RUN_TAG = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# + [markdown] id="dl5KSrInfIGI"
# #### Functions


# + colab={"background_save": true} id="QaaLjXabmhWA"
#######################     Functions     ######################
# =============================================================================
# Function: make_batch
# Purpose:
#   Splits a shapefile (GeoDataFrame) into smaller batches for parallel or
#   distributed processing, based on unique field identifiers.
#
# Parameters:
#   shapfile     → GeoDataFrame containing a column named 'pointID'
#   num_batches  → Total number of batches to split the shapefile into
#   batch_number → Index (1-based) of the specific batch to create
#
# Returns:
#   shapefile_batch → A new GeoDataFrame containing only the subset of
#                     features (pointIDs) for the given batch
# =============================================================================
def make_batch(shapfile, num_batches, batch_number):
    # Extract all unique point identifiers from the shapefile
    unique_pointIDs = shapfile["pointID"].unique()

    # Count how many unique fields/points exist
    field_count = len(unique_pointIDs)
    print(f"{field_count = }")

    # Calculate how many pointIDs should go into each batch
    # Integer division ensures equal-sized batches (except possibly the last one)
    batch_size = field_count // num_batches

    # Compute the index boundaries for the current batch
    start_idx = (batch_number - 1) * batch_size
    if batch_number < num_batches:
        end_idx = start_idx + batch_size
    else:
        # For the last batch, include all remaining pointIDs
        end_idx = field_count

    # Select only the pointIDs that belong to this batch
    current_pointIDs = unique_pointIDs[start_idx:end_idx]

    # Subset the original shapefile to include only these pointIDs
    shapefile_batch = shapfile[shapfile["pointID"].isin(current_pointIDs)].copy()

    # Reset the index to maintain sequential order and avoid inherited indices
    shapefile_batch.reset_index(drop=True, inplace=True)

    # Print the shape (rows, columns) of the resulting batch for verification
    print("shapefile_batch.shape = ", shapefile_batch.shape)

    # Return the subset GeoDataFrame corresponding to this batch
    return shapefile_batch


# =============================================================================
# Function: renameBandsL8
# Purpose:
#   Renames the original band names of Landsat 8 Surface Reflectance (SR) imagery
#   to standardized, human-readable band names for consistent downstream use.
#
# Parameters:
#   image → Earth Engine Image object representing Landsat 8 data
#
# Returns:
#   image → Earth Engine Image with renamed spectral bands
# =============================================================================
def renameBandsL8(image):
    # Original Landsat 8 band names (Surface Reflectance)
    bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL"]

    # New, standardized names (Blue, Green, Red, NIR, SWIR1, SWIR2, QA)
    new_bands = ["B", "G", "R", "NIR", "SWIR1", "SWIR2", "QA_PIXEL"]

    # Select and rename the bands accordingly
    return image.select(bands).rename(new_bands)


# =============================================================================
# Function: renameBandsL7_and_5
# Purpose:
#   Renames the band names of Landsat 7 and Landsat 5 SR imagery to match the
#   same standardized scheme used for Landsat 8, ensuring uniformity.
#
# Parameters:
#   image → Earth Engine Image object (Landsat 5 or 7 SR data)
#
# Returns:
#   image → Earth Engine Image with renamed bands
# =============================================================================
def renameBandsL7_and_5(image):
    # Original Landsat 5/7 SR band names
    bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"]

    # New standardized band names matching Landsat 8 naming scheme
    new_bands = ["B", "G", "R", "NIR", "SWIR1", "SWIR2", "QA_PIXEL"]

    # Select and rename the bands accordingly
    return image.select(bands).rename(new_bands)


# =============================================================================
# Function: applyScaleFactors
# Purpose:
#   Applies official scaling factors to convert Landsat Collection 2 Level 2
#   Surface Reflectance (SR) and Surface Temperature (ST) digital numbers
#   into physically meaningful reflectance and temperature values.
#
# Parameters:
#   image → Earth Engine Image object containing SR and ST bands
#
# Returns:
#   image → Image with scaled optical and thermal bands added (original bands
#            are replaced with scaled ones)
# =============================================================================
def applyScaleFactors(image):
    # Scale optical reflectance bands (Surface Reflectance)
    # Expression: Reflectance = SR_DN * 0.0000275 + (-0.2)
    opticalBands = image.select("SR_B.").multiply(0.0000275).add(-0.2)

    # Scale surface temperature bands (Surface Temperature)
    # Expression: Temperature = ST_DN * 0.00341802 + 149.0
    thermalBands = image.select("ST_B.*").multiply(0.00341802).add(149.0)

    # Note: In this workflow, thermal bands are not used but are scaled for completeness

    # Add scaled optical and thermal bands back to the image,
    # overwriting the original unscaled versions
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)


# =============================================================================
# Function: addIndices
# Purpose:
#   Compute a set of spectral indices from an image that already contains
#   standardized bands: B, G, R, NIR, SWIR1, SWIR2.
#
# Parameters:
#   image → ee.Image with bands ["B","G","R","NIR","SWIR1","SWIR2"]
#
# Returns:
#   ee.Image → input image with added bands:
#              ["evi","gcvi","sndvi","ndti","ndi5","ndi7","crc","sti"]
# =============================================================================
def addIndices(image):
    evi = image.expression(
        '2.5 * (b("NIR") - b("R"))/(b("NIR") + 6 * b("R") - 7.5 * b("B") + 1)'
    ).rename(
        "evi"
    )  # Enhanced Vegetation Index

    gcvi = image.expression('b("NIR")/b("G") - 1').rename(
        "gcvi"
    )  # Green Chlorophyll Vegetation Index

    sndvi = image.expression('(b("NIR") - b("R"))/(b("NIR") + b("R") + 0.16)').rename(
        "sndvi"
    )  # standardized normalized difference vegetation index

    ndti = image.expression(
        '(b("SWIR1") - b("SWIR2"))/(b("SWIR1") + b("SWIR2"))'
    ).rename(
        "ndti"
    )  # Normalized Difference Tillage Index

    ndi5 = image.expression('(b("NIR") - b("SWIR1"))/(b("NIR") + b("SWIR1"))').rename(
        "ndi5"
    )  # NIR–SWIR1 normalized difference

    ndi7 = image.expression('(b("NIR") - b("SWIR2"))/(b("NIR") + b("SWIR2"))').rename(
        "ndi7"
    )  # NIR–SWIR2 normalized difference

    crc = image.expression('(b("SWIR1") - b("G"))/(b("SWIR1") + b("G"))').rename(
        "crc"
    )  # Crop Residue Cover index

    sti = image.expression('b("SWIR1")/b("SWIR2")').rename(
        "sti"
    )  # Simple Tillage Index

    return (
        image.addBands(evi)
        .addBands(gcvi)
        .addBands(sndvi)
        .addBands(ndti)
        .addBands(ndi5)
        .addBands(ndi7)
        .addBands(crc)
        .addBands(sti)
    )


# =============================================================================
# Function: cloudMaskL8
# Purpose:
#   Build a combined QA-based mask for Landsat 8/9 Collection 2 L2 images using
#   QA_PIXEL bits and apply it to the image, preserving acquisition time.
#
# Parameters:
#   image → ee.Image with band "QA_PIXEL"
#
# Returns:
#   ee.Image → masked image (cirrus, cloud, cloud shadow, snow, high cloud conf)
# =============================================================================


def cloudMaskL8(image):
    qa = image.select("QA_PIXEL")

    # Single-bit flags: keep pixels where bit == 0
    cirrus_clear = qa.bitwiseAnd(1 << 2).eq(0)  # bit 2: cirrus
    cloud_clear = qa.bitwiseAnd(1 << 3).eq(0)  # bit 3: cloud
    shadow_clear = qa.bitwiseAnd(1 << 4).eq(0)  # bit 4: cloud shadow
    snow_clear = qa.bitwiseAnd(1 << 5).eq(0)  # bit 5: snow

    # Cloud confidence (bits 8–9): 0=None, 1=Low, 2=Medium, 3=High
    cloud_conf = qa.rightShift(8).bitwiseAnd(3)
    not_high_conf = cloud_conf.neq(3)  # Drop high-confidence cloudy pixels

    # Preserve existing valid pixels
    keep_existing = image.mask().reduce(ee.Reducer.min())

    return (
        image.updateMask(cirrus_clear)
        .updateMask(cloud_clear)
        .updateMask(shadow_clear)
        .updateMask(snow_clear)
        .updateMask(not_high_conf)
        .updateMask(keep_existing)
        .copyProperties(image, ["system:time_start"])
    )


# =============================================================================
# Function: addNDVI
# Purpose:
#   Add the NDVI band computed from NIR and R.
#
# Parameters:
#   image → ee.Image with bands ["NIR", "R"]
#
# Returns:
#   ee.Image → input image with an added "ndvi" band
# =============================================================================
def addNDVI(image):
    ndvi = image.normalizedDifference(["NIR", "R"]).rename("ndvi")
    return image.addBands(ndvi)


# =============================================================================
# Function: maskNDVI
# Purpose:
#   Apply a mask to the image keeping only pixels where NDVI ≤ threshold.
#   (Useful for excluding vegetation above a certain greenness.)
#
# Parameters:
#   image     → ee.Image that already contains an "ndvi" band
#   threshold → numeric threshold; pixels with NDVI > threshold are masked out
#
# Returns:
#   ee.Image → masked image
# =============================================================================
def maskNDVI(image, threshold):
    NDVI = image.select("ndvi")  # select NDVI band
    ndviMask = NDVI.lte(threshold)  # boolean mask: ndvi <= threshold
    masked = image.updateMask(ndviMask)  # apply mask
    return masked


# =============================================================================
# Function: MoistMask
# Purpose:
#   Mask pixels if the 3-day cumulative precipitation from GridMET exceeds 3 mm.
#   Also attaches the 3-day sum as band 'pr_sum' to the image.
#
# Parameters:
#   img      → ee.Image (the target image to mask)
#   GridMet  → ee.ImageCollection with a 'pr' band (daily precipitation)
#
# Returns:
#   ee.Image → masked image with added 'pr_sum' band
# =============================================================================
def MoistMask(img, GridMet):
    """Mask image where 3-day cumulative GridMET precipitation > 3 mm; adds 'pr_sum' band."""
    date_0 = img.date()  # current image date
    date_next = date_0.advance(+1, "day")  # upper bound (exclusive)
    date_1 = date_0.advance(-1, "day")  # previous day
    date_2 = date_0.advance(-2, "day")  # two days prior

    Gimg1 = GridMet.filterDate(
        date_2, date_1
    )  # day -2 (inclusive) to day -1 (exclusive)
    Gimg2 = GridMet.filterDate(date_1, date_0)  # day -1 to day 0
    Gimg3 = GridMet.filterDate(date_0, date_next)  # day 0 to day +1

    GridMColl_123 = ee.ImageCollection(Gimg1.merge(Gimg2).merge(Gimg3))  # concat 3 days
    GridMetImgpr = GridMColl_123.select("pr")  # precipitation band
    threeDayPrec = GridMetImgpr.reduce(ee.Reducer.sum())  # sum → 'pr_sum'

    img = img.addBands(threeDayPrec)  # attach pr_sum
    MaskedGMImg = threeDayPrec.lte(3).select("pr_sum").eq(1)  # keep <= 3 mm
    maskedLImg = img.updateMask(MaskedGMImg)  # apply mask
    return maskedLImg


# =============================================================================
# Function: addDOY
# Purpose:
#   Add a constant band 'doy' (0-based day-of-year) to each image and set a
#   human-readable date property 'doy_date'.
#
# Parameters:
#   img → ee.Image with 'system:time_start'
#
# Returns:
#   ee.Image → input image with 'doy' band and 'doy_date' property
# =============================================================================
def addDOY(img):
    """
    Add a constant 'doy' band (0-based day-of-year) and a 'doy_date' (YYYY-MM-dd) property.
    Parameters:
      img: ee.Image with a valid system:time_start
    Returns:
      ee.Image with an added 'doy' band (uint16) and 'doy_date' property.
    """
    doy = img.date().getRelative("day", "year")  # 0..364/365
    doyBand = ee.Image.constant(doy).uint16().rename("doy")  # constant band
    date = img.date().format("YYYY-MM-dd")  # formatted date string
    img = img.addBands(doyBand)  # add band
    img = img.set("doy_date", date)  # set property
    return img


# =============================================================================
# Function: groupImages
# Purpose:
#   From an original ImageCollection, build two seasonal collections for a given
#   year and rename all selected bands with season-specific suffixes.
#   - Season 0 (S0): Sep 1 – Dec 30 of the given year
#   - Season 1 (S1): Mar 1 – May 30 of the next year
#
# Parameters:
#   year          → int or ee.Number (base year)
#   orgCollection → ee.ImageCollection (already prepared with required bands)
#   geometry      → ee.Geometry (spatial filter/clip)
#
# Returns:
#   [collection_0, collection_1] → list of ee.ImageCollection
# =============================================================================
def groupImages(year, orgCollection, geometry):
    """
    Group an Earth Engine ImageCollection into two seasonal sub-collections for a given year
    and rename bands with season-specific suffixes.

    - S0: Fall window of the given year (Sep 1 – Nov 30)
    - S1: Spring window of the next year (Mar 1 – May 30)

    Returns:
      [collection_0, collection_1]
    """
    bands = [
        "B",
        "G",
        "R",
        "NIR",
        "SWIR1",
        "SWIR2",
        "evi",
        "gcvi",
        "ndvi",
        "sndvi",
        "ndti",
        "ndi5",
        "ndi7",
        "crc",
        "sti",
        "doy",
    ]  # expected input bands

    new_bandS0 = [
        "B_S0",
        "G_S0",
        "R_S0",
        "NIR_S0",
        "SWIR1_S0",
        "SWIR2_S0",
        "evi_S0",
        "gcvi_S0",
        "ndvi_S0",
        "sndvi_S0",
        "ndti_S0",
        "ndi5_S0",
        "ndi7_S0",
        "crc_S0",
        "sti_S0",
        "doy_S0",
    ]  # renamed for Season 0

    new_bandS1 = [
        "B_S1",
        "G_S1",
        "R_S1",
        "NIR_S1",
        "SWIR1_S1",
        "SWIR2_S1",
        "evi_S1",
        "gcvi_S1",
        "ndvi_S1",
        "sndvi_S1",
        "ndti_S1",
        "ndi5_S1",
        "ndi7_S1",
        "crc_S1",
        "sti_S1",
        "doy_S1",
    ]  # renamed for Season 1

    year = ee.Number(year)  # ensure server-side number
    next_year = year.add(1)  # following calendar year

    collection_0 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(year, 9, 1), ee.Date.fromYMD(year, 11, 30)
        )  # S0 window
        .filterBounds(geometry)  # spatial filter
        .map(addDOY)  # add DOY band/property
        .map(lambda img: img.select(bands).rename(new_bandS0))  # rename bands
    )

    collection_1 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(next_year, 3, 1), ee.Date.fromYMD(next_year, 5, 30)
        )  # S1 window
        .filterBounds(geometry)  # spatial filter
        .map(addDOY)  # add DOY band/property
        .map(lambda img: img.select(bands).rename(new_bandS1))  # rename bands
    )

    return [collection_0, collection_1]


# =============================================================================
# Function: percentile_imageReducer
# Purpose:
#   For each image in a list (with potentially different band names), compute
#   the per-feature median over a feature collection (e.g., fields) at 30 m scale.
#
# Parameters:
#   imageList → list of ee.Image (Python list or server-side list)
#   shp       → ee.FeatureCollection (polygons/points to reduce over)
#
# Returns:
#   list[ee.FeatureCollection] → each element corresponds to one image’s
#   reduceRegions() result (median per feature), computed with tileScale=16.
# =============================================================================
def percentile_imageReducer(imageList, shp):
    return list(
        map(
            lambda img: ee.Image(img).reduceRegions(
                **{
                    "reducer": ee.Reducer.median(),
                    "collection": shp,
                    "scale": 30,
                    "tileScale": 16,
                }
            ),
            imageList,
        )
    )


# =============================================================================
# Function: applyGLCM
# Purpose:
#   Compute gray-level co-occurrence matrix (GLCM) texture features for each
#   image in an ImageCollection, preserving acquisition time.
#
# Parameters:
#   coll → ee.ImageCollection
#
# Returns:
#   ee.ImageCollection → each image has GLCM texture bands; 'system:time_start' set
# =============================================================================
def applyGLCM(coll):
    int32Coll = coll.map(
        lambda img: img.toInt32()
    )  # cast all bands to int32 before GLCM
    glcmColl = int32Coll.map(
        lambda img: img.glcmTexture().set(
            "system:time_start", img.date()
        )  # add texture; keep time
    )
    return glcmColl


# =============================================================================
# Function: remove_doy
# Purpose:
#   From a Python list of ee.Image objects, remove any bands whose names start
#   with 'doy' (e.g., 'doy', 'doy_S0', 'doy_S1') and return a new list.
#
# Parameters:
#   image_list → list[ee.Image]
#
# Returns:
#   list[ee.Image] → images with 'doy*' bands removed
# =============================================================================
def remove_doy(image_list):
    new_image_list = []  # accumulator for filtered images
    for img in image_list:  # iterate over ee.Image items
        all_band_names = img.bandNames()  # get server-side list of band names
        bands_to_keep = all_band_names.filter(  # filter out names that start with 'doy'
            ee.Filter.stringStartsWith("item", "doy").Not()
        )
        img_filtered = img.select(bands_to_keep)  # select only kept bands
        new_image_list.append(img_filtered)  # collect result
    return new_image_list


# =============================================================================
# Function: _drive_service
# Purpose:
#   Build an authenticated Google Drive v3 service client using a service
#   account file and the provided OAuth scopes.
#
# Parameters:
#   (uses global SERVICE_ACCOUNT_FILE and SCOPES)
#
# Returns:
#   googleapiclient.discovery.Resource → Drive service
# =============================================================================
def _drive_service():
    creds = service_account.Credentials.from_service_account_file(  # load service account key
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build(
        "drive", "v3", credentials=creds, cache_discovery=False
    )  # construct client


# =============================================================================
# Function: _start_export_to_drive
# Purpose:
#   Kick off an Earth Engine table (FeatureCollection) export to Google Drive
#   as a CSV file.
#
# Parameters:
#   fc           → ee.FeatureCollection
#   description  → str, task description
#   file_prefix  → str, Drive filename prefix (without extension)
#   folder       → Optional[str], Drive folder name (None → root)
#
# Returns:
#   ee.batch.Task → started EE export task
# =============================================================================
def _start_export_to_drive(fc, description, file_prefix, folder=None):
    """Start an EE toDrive export job."""
    task = ee.batch.Export.table.toDrive(  # configure table export
        collection=fc,
        description=description,
        folder=folder,  # None → root folder
        fileNamePrefix=file_prefix,  # filename prefix
        fileFormat="CSV",  # CSV output
    )
    task.start()  # start async task
    return task


# =============================================================================
# Function: _wait_for_task
# Purpose:
#   Poll an Earth Engine task until it finishes; raise an error if it fails.
#
# Parameters:
#   task          → ee.batch.Task
#   poll_seconds  → int, seconds to sleep between polls
#
# Returns:
#   dict → task status on completion (state == 'COMPLETED' or raises)
# =============================================================================
def _wait_for_task(task, poll_seconds=15):
    """Block until task finishes; raise RuntimeError on failure."""
    while task.active():  # keep polling while task is active
        time.sleep(poll_seconds)  # wait between checks
    status = task.status()  # fetch final status
    state = status.get("state", "UNKNOWN")
    if state != "COMPLETED":  # treat anything else as failure
        raise RuntimeError(f"EE export failed: {status}")
    return status


# =============================================================================
# Function: _find_latest_csv_by_name
# Purpose:
#   In Google Drive, find the most recently created CSV file that matches an
#   exact filename (e.g., 'prefix.csv'). Handles duplicates by creation time.
#
# Parameters:
#   service   → Drive service client
#   filename  → str, exact name to match (including '.csv')
#
# Returns:
#   Optional[str] → fileId of newest match, or None if not found
# =============================================================================
def _find_latest_csv_by_name(service, filename):
    """
    Return the most recent fileId for an exact Drive filename (e.g., 'name.csv').
    Drive can hold duplicates; we pick the latest created.
    """
    query = (
        f"name = '{filename}' and mimeType = 'text/csv'"  # exact name + CSV mimetype
    )
    resp = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            orderBy="createdTime desc",  # newest first
            fields="files(id, name, createdTime)",  # only what we need
            pageSize=5,  # small page is enough
        )
        .execute()
    )
    items = resp.get("files", [])  # list of file dicts
    return items[0]["id"] if items else None  # top match or None


# =============================================================================
# Function: _download_csv_to_dataframe
# Purpose:
#   Given a Drive fileId for a CSV file, download it into memory and parse as
#   a pandas DataFrame.
#
# Parameters:
#   service  → Drive service client
#   file_id  → str, Drive fileId
#
# Returns:
#   pandas.DataFrame
# =============================================================================
def _download_csv_to_dataframe(service, file_id):
    """Download Drive fileId to pandas DataFrame (in-memory)."""
    request = service.files().get_media(fileId=file_id)  # media request handle
    fh = io.BytesIO()  # in-memory buffer
    downloader = MediaIoBaseDownload(fh, request)  # chunked downloader
    done = False
    while not done:  # iterate until finished
        _, done = downloader.next_chunk()
    fh.seek(0)  # rewind to start
    return pd.read_csv(fh)  # parse CSV


# =============================================================================
# Function: _delete_drive_file
# Purpose:
#   Best-effort deletion of a Drive file by fileId; errors are suppressed.
#
# Parameters:
#   service  → Drive service client
#   file_id  → str, Drive fileId
#
# Returns:
#   None
# =============================================================================
def _delete_drive_file(service, file_id):
    try:
        service.files().delete(fileId=file_id).execute()  # attempt deletion
    except Exception:
        pass  # ignore failures


# =============================================================================
# Function: export_to_drive_and_read
# Purpose:
#   For each (seasonal) FeatureCollection in a year-ordered list, export to
#   Google Drive as CSV, wait for completion, download to pandas, and return
#   both per-year lists and a combined DataFrame.
#
# Parameters:
#   feature_type           → str, label for filenames (e.g., 'main' or 'glcm')
#   reduced_list           → list of [S0_fc, S1_fc] per year (ordered by years)
#   Bands                  → list[str], band names to include in export
#   years                  → list[int], base years aligned with reduced_list
#   run_tag                → str, unique tag to avoid collisions (default RUN_TAG)
#   folder                 → Optional[str], Drive folder name (None → root)
#   important_cols         → Optional[list[str]], extra non-band columns to keep
#   sleep_between_starts   → float, throttle between task starts (seconds)
#
# Returns:
#   (yearly_df_list, combined_df)
#     yearly_df_list → [[df_y_S0, df_y_S1], ...]
#     combined_df    → concatenated DataFrame with 'season' column
# =============================================================================
def export_to_drive_and_read(
    feature_type,
    reduced_list,
    Bands,
    years,
    *,
    run_tag=RUN_TAG,
    folder=FOLDER,
    important_cols=None,
    sleep_between_starts=0.8,
):
    """
    feature_type: string ('main', 'glcm')
    reduced_list: [[S0_fc, S1_fc], [S0_fc, S1_fc], ...] ordered by year
    Bands: list of band names you exported
    startYear: int (e.g., 2021)
    folder: None => Drive root (recommended to avoid duplicate-folder issue)
    Returns:
      yearly_df_list: [[df_y_S0, df_y_S1], ...]
      and a combined DataFrame with an added 'season' column.
    """
    if important_cols is None:
        important_cols = ["pointID"]  # default minimal identifier set

    service = _drive_service()  # authenticated Drive client
    yearly_df_list = []  # per-year list of [S0_df, S1_df]
    all_frames = []  # accumulator for combined output

    for i, year_item in enumerate(reduced_list):  # iterate over years
        year = years[i] + 1  # filenames use next-year label
        season_frames = []  # holder for S0/S1 dataframes
        for s, fc in enumerate(year_item):  # season index s ∈ {0,1}
            selectors = important_cols + Bands  # restrict exported properties
            fc_sel = ee.FeatureCollection(fc).select(
                selectors
            )  # select only needed columns

            file_prefix = f"{run_tag}-{feature_type}_y{year}_S{s}"  # unique filename
            description = file_prefix  # task description

            task = _start_export_to_drive(  # start export
                fc_sel, description, file_prefix, folder=folder
            )
            time.sleep(sleep_between_starts)  # throttle task starts

            _wait_for_task(task)  # block until completed

            csv_name = f"{file_prefix}.csv"  # expected Drive name
            file_id = _find_latest_csv_by_name(service, csv_name)  # locate the file
            if not file_id:
                raise FileNotFoundError(f"Could not find Drive file named {csv_name}")

            df = _download_csv_to_dataframe(service, file_id)  # read into pandas
            _delete_drive_file(service, file_id)  # cleanup Drive file

            df.insert(1, "season", f"S{s}")  # annotate season column
            season_frames.append(df)  # collect season df
            all_frames.append(df)  # collect for combined

        yearly_df_list.append(season_frames)  # append year pair

    combined_df = (
        pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    )  # build combined output
    return yearly_df_list, combined_df


# =============================================================================
# Function: extract_terrain
# Purpose:
#   Compute zonal mean terrain attributes (elevation, slope, aspect) over each
#   feature in one or more FeatureCollections and return a pandas DataFrame.
#
# Parameters:
#   shapefile_list → list[ee.FeatureCollection], each with id_field property
#   id_field       → str, unique identifier property name (default 'pointID')
#   scale          → int, pixel scale in meters for reduction (default 30)
#   tilescale      → int, tileScale for EE reducers to manage memory (default 4)
#
# Returns:
#   pandas.DataFrame → columns [id_field, 'elevation', 'slope', 'aspect']
# =============================================================================
def extract_terrain(
    shapefile_list, id_field="pointID", scale=30, tilescale=4
) -> pd.DataFrame:
    dem = (
        ee.Image("NASA/NASADEM_HGT/001").select("elevation").rename("elevation")
    )  # elevation image
    slope = ee.Terrain.slope(dem).rename("slope")  # slope (degrees)
    aspect = ee.Terrain.aspect(dem).rename("aspect")  # aspect (degrees)
    terrain = dem.addBands([slope, aspect])  # stack into one image

    def zonal_stats_mean(fc, id_field=id_field, scale=scale, tilescale=tilescale):
        fc_id = fc.map(
            lambda f: f.select([id_field]).copyProperties(f, [id_field])
        )  # retain only ID property
        stats = terrain.reduceRegions(  # per-feature mean reducer
            collection=fc_id,
            reducer=ee.Reducer.mean(),
            scale=scale,
            crs=dem.projection(),
            tileScale=tilescale,
        )
        stats = stats.map(  # keep explicit fields
            lambda f: f.select(
                [id_field, "elevation", "slope", "aspect"],
                [id_field, "elevation", "slope", "aspect"],
            )
        )
        return stats

    fc_stats_list = [  # compute stats for each FC
        zonal_stats_mean(fc, id_field=id_field, scale=scale, tilescale=tilescale)
        for fc in shapefile_list
    ]
    merged_stats_fc = (  # merge and select fields
        ee.FeatureCollection(fc_stats_list)
        .flatten()
        .select([id_field, "elevation", "slope", "aspect"])
    )

    df = geemap.ee_to_df(merged_stats_fc)  # convert to pandas
    df = df[[id_field, "elevation", "slope", "aspect"]].reset_index(
        drop=True
    )  # order columns
    return df


# =============================================================================
# Function: addYear
# Purpose:
#   Add a 'year' property to an ee.Image using its 'system:time_start'.
#
# Parameters:
#   image → ee.Image with 'system:time_start'
#
# Returns:
#   ee.Image → input image with an added 'year' property
# =============================================================================
def addYear(image):
    return image.set("year", ee.Date(image.get("system:time_start")).get("year"))


# + [markdown] id="DUhdHR8xIrUE"
# # Download Metric-Based Landsat Data

# + colab={"background_save": true} id="vrRY7E6NLhul"
# =============================================================================
# Study Area Geometry and Local Paths
# Purpose:
#   Define analysis geometries (polygon + example point) and configure local
#   filesystem paths used to locate shapefiles.
# =============================================================================

geometry = ee.Geometry.Polygon(
    [
        [
            [-118.61039904725511, 47.40441980731236],
            [-118.61039904725511, 45.934467488469],
            [-116.80864123475511, 45.934467488469],
            [-116.80864123475511, 47.40441980731236],
        ]
    ],
    None,
    False,
)  # main AOI polygon


# Root folder for project data (replace with your own path)
path_to_data = "/path/to/your/tillage_mapping_data_root/"
shapefile_dir = os.path.join(
    path_to_data, "final_shpfiles/"
)  # directory containing .shp files


# =============================================================================
# Batch Parameters
# Purpose:
#   Configure batching for per-field processing (e.g., parallelization).
#   num_batches: total splits; batch_number: 1-based index of the split to run.
# =============================================================================

# These are hard coded here based on the number of fields in the shapefile.
num_batches = int(sys.argv[1])
batch_number = int(sys.argv[2])


# =============================================================================
# Shapefile Discovery and Loading
# Purpose:
#   Discover yearly shapefiles, sort by the year embedded in filename, and
#   build a dict {year → GeoDataFrame} with minimal columns for efficiency.
# Assumptions:
#   Filenames contain a 4-digit year matching regex (20\d{2}).
# =============================================================================

shapefiles = sorted(
    [f for f in os.listdir(shapefile_dir) if f.endswith(".shp")],
    key=lambda s: int(re.search(r"(20\d{2})", s).group(1)),  # sort by detected year
)

shp_by_year = {
    int(re.search(r"(20\d{2})", f).group(1)): gpd.read_file(
        shapefile_dir + f,
        engine="pyogrio",  # fast driver
        usecols=["geometry", "pointID"],  # only what downstream needs
    )
    for f in shapefiles
}


# =============================================================================
# Target Years and Shapefile Alignment
# Purpose:
#   Choose imagery base years and align per-field shapefiles to image years
#   using the "year + 1" rule (labels the spring of next year).
# =============================================================================

years = [2009, 2019, 2023]  # imagery base years
shpfiles = [shp_by_year[y + 1] for y in years]  # align to y+1 shapefile


# =============================================================================
# Batch Creation and EE Conversion
# Purpose:
#   Split each year's GeoDataFrame into the requested batch and convert each
#   to an Earth Engine FeatureCollection for server-side processing.
# =============================================================================

geopandas_list = [
    make_batch(shp, num_batches, batch_number) for shp in shpfiles
]  # subset per batch
shpfilesList = [
    geemap.geopandas_to_ee(shp) for shp in geopandas_list
]  # convert to ee.FeatureCollection


# =============================================================================
# Expanded Year Window
# Purpose:
#   Create a deduplicated, sorted list including (year - 1), year, (year + 1)
#   to support fall (Sep–Dec, year i) and spring (Mar–May, year i+1) windows.
# =============================================================================

expanded_years = sorted(
    list(set([year - 1 for year in years] + years + [year + 1 for year in years]))
)


# =============================================================================
# Landsat Collections (C02 L2)
# Purpose:
#   Load Landsat 5/7/8 Tier 1 L2 collections, annotate with a 'year' property,
#   constrain to the expanded year window, and spatially clip to the AOI.
# Notes:
#   The .map(addYear) extracts 'year' from 'system:time_start', then we set 'year'
#   again from img.date() for redundancy, preserving original behavior exactly.
# =============================================================================

L8T1 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
L7T1 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
L5T1 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")

L8 = (
    L8T1.map(addYear)  # add 'year' property from time_start
    .filter(ee.Filter.inList("year", expanded_years))  # keep only expanded-year images
    .map(
        lambda img: img.set("year", img.date().get("year"))
    )  # set 'year' (redundant but kept)
    .map(lambda img: img.clip(geometry))  # clip to AOI polygon
)

L7 = (
    L7T1.map(addYear)
    .filter(ee.Filter.inList("year", expanded_years))
    .map(lambda img: img.set("year", img.date().get("year")))
    .map(lambda img: img.clip(geometry))
)

L5 = (
    L5T1.map(addYear)
    .filter(ee.Filter.inList("year", expanded_years))
    .map(lambda img: img.set("year", img.date().get("year")))
    .map(lambda img: img.clip(geometry))
)


# =============================================================================
# Radiometric Scaling and Band Standardization
# Purpose:
#   Apply Collection 2 Level-2 scaling factors, then harmonize band names so
#   all missions share a common schema for downstream indices and reductions.
# =============================================================================

L8 = L8.map(applyScaleFactors)  # convert SR/ST DNs to reflectance/temperature
L7 = L7.map(applyScaleFactors)
L5 = L5.map(applyScaleFactors)

L8 = L8.map(
    renameBandsL8
)  # standardize to ["B","G","R","NIR","SWIR1","SWIR2","QA_PIXEL"]
L7 = L7.map(renameBandsL7_and_5)
L5 = L5.map(renameBandsL7_and_5)


# =============================================================================
# Merge, Cloud Masking, and Vegetation Masking
# Purpose:
#   Combine missions, apply QA-based cloud/snow/shadow/cirrus masking, compute
#   NDVI, and then mask pixels above a greenness threshold.
#   (Threshold of 0.2 retained exactly.)
# =============================================================================

landSat_5_7_8 = ee.ImageCollection(L8.merge(L7).merge(L5))  # merge missions

landSat_5_7_8_masked = landSat_5_7_8.map(cloudMaskL8)  # QA-based masks (bits 2,3,4,5,9)

landSat_5_7_8_masked = landSat_5_7_8_masked.map(addNDVI)  # add NDVI band

landSat_5_7_8_masked = landSat_5_7_8_masked.map(  # keep pixels with NDVI ≤ 0.2
    lambda image: maskNDVI(image, threshold=0.2)
)

# =============================================================================
# GRIDMET Filtering and Moisture Masking
# Purpose:
#   Load GRIDMET, annotate with 'year', restrict by expanded_years and AOI,
#   and apply the 3-day precipitation mask to each Landsat image.
# =============================================================================

GridMet = (
    ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")  # daily GRIDMET collection
    .map(addYear)  # add 'year' from system:time_start
    .filter(ee.Filter.inList("year", expanded_years))  # keep only needed years
    .filterBounds(geometry)  # spatial filter to AOI
)

landSat_5_7_8_masked = landSat_5_7_8_masked.map(  # apply moist mask to each image
    lambda image: MoistMask(image, GridMet)
)


# =============================================================================
# Add Spectral Indices
# Purpose:
#   For each image in the masked collection, compute vegetation/soil indices
#   (evi, gcvi, ndti, etc.) and append them as new bands.
# =============================================================================

landSat_5_7_8_masked = landSat_5_7_8_masked.map(addIndices)

# =============================================================================
# Seasonal Grouping
# Purpose:
#   For each target year, create seasonal ImageCollections:
#   S0 = Sep 1–Nov 30 of year y, S1 = Mar 1–May 30 of year y+1.
# Returns:
#   yearlyCollectionsList: list over years of [S0_collection, S1_collection].
# =============================================================================

yearlyCollectionsList = [groupImages(y, landSat_5_7_8_masked, geometry) for y in years]

# =============================================================================
# Clip to Field Boundaries
# Purpose:
#   Clip every seasonal ImageCollection to the year-matched WSDA field polygons.
# Shape:
#   [[S0_clipped, S1_clipped],  ... per year ... ]
# =============================================================================

clipped_mainBands_CollectionList = list(
    map(
        lambda collList, shp: list(
            map(
                lambda collection: ee.ImageCollection(collection).map(
                    lambda img: ee.Image(img).clip(shp)  # per-image field clip
                ),
                collList,
            )
        ),
        yearlyCollectionsList,
        shpfilesList,
    )
)

# =============================================================================
# GLCM Texture Extraction
# Purpose:
#   For each seasonal collection, compute GLCM texture features on all images.
# Shape mirrors clipped_mainBands_CollectionList.
# =============================================================================

clipped_GLCM_collectionList = list(
    map(
        lambda collList: list(map(applyGLCM, collList)),
        clipped_mainBands_CollectionList,
    )
)


# =============================================================================
# Percentile Composites
# Purpose:
#   Reduce each seasonal ImageCollection to percentile images for a fixed set
#   of percentiles (0,5,25,50,75,100), for both main bands and GLCM bands.
# =============================================================================

percentiles = [0, 5, 25, 50, 75, 100]
mainBands_percentile_collectionList = list(
    map(
        lambda collList: list(
            map(
                lambda collection: collection.reduce(
                    ee.Reducer.percentile(percentiles)  # composite to percentiles
                ),
                collList,
            )
        ),
        clipped_mainBands_CollectionList,
    )
)

mainBands_percentile_collectionList = [
    remove_doy(image_list)
    for image_list in mainBands_percentile_collectionList  # drop 'doy*' bands
]

glcmBands_percentile_collectionList = list(
    map(
        lambda collList: list(
            map(
                lambda collection: collection.reduce(
                    ee.Reducer.percentile(percentiles)
                ),
                collList,
            )
        ),
        clipped_GLCM_collectionList,
    )
)

glcmBands_percentile_collectionList = [
    remove_doy(image_list) for image_list in glcmBands_percentile_collectionList
]


# =============================================================================
# Zonal Reduction to Fields (Main and GLCM)
# Purpose:
#   For each percentile image, compute median over each field geometry (30 m).
# Returns:
#   reducedList_*: per year → list of FeatureCollections (one per season).
# =============================================================================

reducedList_mainBands = list(
    map(
        lambda imgList, shp: percentile_imageReducer(
            imgList, ee.FeatureCollection(shp)
        ),
        mainBands_percentile_collectionList,
        shpfilesList,
    )
)

reducedList_glcmBands = list(
    map(
        lambda imgList, shp: percentile_imageReducer(
            imgList, ee.FeatureCollection(shp)
        ),
        glcmBands_percentile_collectionList,
        shpfilesList,
    )
)


# =============================================================================
# Band Name Extraction
# Purpose:
#   Read band names from the first year's composites and flatten into a list.
#   (Assumes identical schema across years.)
# =============================================================================

nameLists = list(
    map(lambda img: img.bandNames().getInfo(), mainBands_percentile_collectionList[0])
)
mainBands = [name for sublist in nameLists for name in sublist]

nameLists = list(
    map(
        lambda img: ee.Image(img).bandNames().getInfo(),
        glcmBands_percentile_collectionList[0],
    )
)
glcmBands = [name for sublist in nameLists for name in sublist]


# =============================================================================
# Export, Read Back, and Assemble Yearly DataFrames
# Purpose:
#   Export reduced FeatureCollections to Drive (CSV), read to pandas, and
#   produce per-year DataFrames for main and GLCM bands. Then merge per year.
# =============================================================================

important_columns_names = ["pointID"]

yearly_list_mainBands, _ = export_to_drive_and_read(
    "main",
    reducedList_mainBands,
    mainBands,
    years,
    folder=None,  # Drive root
)

yearly_list_glcmBands, _ = export_to_drive_and_read(
    "glcm",
    reducedList_glcmBands,
    glcmBands,
    years,
    folder=None,  # Drive root
)

year_data_list = []
for i, year in enumerate(years):
    year_main = pd.concat(yearly_list_mainBands[i], axis=1)  # concat S0/S1 horizontally
    year_glcm = pd.concat(yearly_list_glcmBands[i], axis=1)  # concat S0/S1 horizontally
    year_data = pd.concat([year_main, year_glcm], axis=1)  # combine main + glcm
    year_data["year"] = f"{year}-{year+1}"  # label season span
    year_data_list.append(year_data)

metric_based_data = pd.concat(year_data_list)  # stack all years


# =============================================================================
# Deduplicate, Terrain Merge, and Column Ordering
# Purpose:
#   Remove duplicate columns, join terrain stats by pointID, and reorder to
#   prioritize identifiers, year label, main/GLCM bands, terrain, and geometry.
# =============================================================================

metric_based_data = metric_based_data.loc[:, ~metric_based_data.columns.duplicated()]

terrain_df = extract_terrain(shpfilesList, id_field="pointID", scale=30, tilescale=4)

metric_based_data = pd.merge(metric_based_data, terrain_df, on="pointID", how="right")

terrain_cols = ["elevation", "slope", "aspect"]

important_columns = (
    important_columns_names + ["year"] + mainBands + glcmBands + terrain_cols + [".geo"]
)

metric_based_data = metric_based_data[important_columns]
metric_based_data = metric_based_data.sort_values(by="pointID")


# =============================================================================
# Save Batch Output
# Purpose:
#   Write the assembled per-batch DataFrame to CSV using the batch_number in
#   the filename. (Directory must exist beforehand.)
# =============================================================================

path_to_save = "path_to_save_batches/"  # output directory
metric_based_data.to_csv(path_to_save + f"map_data_batch_{batch_number}")
