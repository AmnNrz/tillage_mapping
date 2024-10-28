# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + id="whtD--m_FObX"
# Initialize GEE python API
import ee

# Trigger the authentication flow.
ee.Authenticate()
# Initialize the library.
ee.Initialize()

# # Install geemap
# # !pip install geemap
# # !pip install geopandas
import ee
import geemap
import numpy as np
import geopandas as gpd
import pandas as pd
import time
import os
import json

# # Mount google drive
# from google.colab import drive
# drive.mount('/content/drive')

# +
import geopandas as gpd
path_to_data = "/Users/aminnorouzi/Library/CloudStorage/OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/Projects/Tillage_Mapping/Data/field_level_data/mapping_data/2017/"

df = gpd.read_file(path_to_data + "WSDA_2017.shp")
df.head(4)
# -

whtmn = df.loc[df['County'] == "Whitman"]
clmbia = df.loc[df["County"] == "Columbia"]
whtmn.Acres.sum(), clmbia.Acres.sum()

# + [markdown] id="dl5KSrInfIGI"
# #### Functions

# + colab={"background_save": true} id="QaaLjXabmhWA"
#######################     Functions     ######################

# ///// Rename Landsat 8, 7 and 5 bands /////


def renameBandsL8(image):
    bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL"]
    new_bands = ["B", "G", "R", "NIR", "SWIR1", "SWIR2", "QA_PIXEL"]
    return image.select(bands).rename(new_bands)


def renameBandsL7(image):
    bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"]
    new_bands = ["B", "G", "R", "NIR", "SWIR1", "SWIR2", "QA_PIXEL"]
    return image.select(bands).rename(new_bands)


# ///// Apply scaling factor /////
def applyScaleFactors(image):
    opticalBands = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    thermalBands = (
        image.select("ST_B.*").multiply(0.00341802).add(149.0)
    )  # We are not using thermal bands.
    return image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)


# ///// Computes spectral indices,  including EVI, GCVI, NDVI, SNDVI, NDTI, NDI5, NDI7, CRC, STI
# and adds them as bands to each image /////
def addIndices(image):
    # evi
    evi = image.expression(
        '2.5 * (b("NIR") - b("R"))/(b("NIR") + 6 * b("R") - 7.5 * b("B") + 1)'
    ).rename("evi")

    # gcvi
    gcvi = image.expression('b("NIR")/b("G") - 1').rename("gcvi")

    # sndvi
    sndvi = image.expression('(b("NIR") - b("R"))/(b("NIR") + b("R") + 0.16)').rename(
        "sndvi"
    )

    # ndti
    ndti = image.expression(
        '(b("SWIR1") - b("SWIR2"))/(b("SWIR1") + b("SWIR2"))'
    ).rename("ndti")

    # ndi5
    ndi5 = image.expression('(b("NIR") - b("SWIR1"))/(b("NIR") + b("SWIR1"))').rename(
        "ndi5"
    )

    # ndi7
    ndi7 = image.expression('(b("NIR") - b("SWIR2"))/(b("NIR") + b("SWIR2"))').rename(
        "ndi7"
    )

    # crc
    crc = image.expression('(b("SWIR1") - b("G"))/(b("SWIR1") + b("G"))').rename("crc")

    # sti
    sti = image.expression('b("SWIR1")/b("SWIR2")').rename("sti")

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


# Mask cloud
def cloudMaskL8(image):
    qa = image.select("QA_PIXEL")  ##substitiu a band FMASK
    cloud1 = qa.bitwiseAnd(1 << 3).eq(0)
    cloud2 = qa.bitwiseAnd(1 << 9).eq(0)
    cloud3 = qa.bitwiseAnd(1 << 4).eq(0)

    mask2 = image.mask().reduce(ee.Reducer.min())
    return (
        image.updateMask(cloud1)
        .updateMask(cloud2)
        .updateMask(cloud3)
        .updateMask(mask2)
        .copyProperties(image, ["system:time_start"])
    )


# ///// Add NDVI /////
def addNDVI(image):
    ndvi = image.normalizedDifference(["NIR", "R"]).rename("ndvi")
    return image.addBands(ndvi)


# ///// Mask NDVI /////
def maskNDVI(image, threshold):
    NDVI = image.select("ndvi")
    ndviMask = NDVI.lte(threshold)
    masked = image.updateMask(ndviMask)
    return masked


# ///// Mask pr>0.3 from GridMet image /////
def MoistMask(img, GridMet):
    # Find dates (2 days Prior) and filter Grid collection
    date_0 = img.date()
    date_next = date_0.advance(+1, "day")
    date_1 = date_0.advance(-1, "day")
    date_2 = date_0.advance(-2, "day")
    Gimg1 = GridMet.filterDate(date_2, date_1)
    Gimg2 = GridMet.filterDate(date_1, date_0)
    Gimg3 = GridMet.filterDate(date_0, date_next)

    # Sum of precipitation for all three dates
    GridMColl_123 = ee.ImageCollection(Gimg1.merge(Gimg2).merge(Gimg3))
    GridMetImgpr = GridMColl_123.select("pr")
    threeDayPrec = GridMetImgpr.reduce(ee.Reducer.sum())

    # Add threeDayPrec as a property to the image in the imageCollection
    img = img.addBands(threeDayPrec)
    # mask gridmet image for pr > 3mm
    MaskedGMImg = threeDayPrec.lte(3).select("pr_sum").eq(1)
    maskedLImg = img.updateMask(MaskedGMImg)
    return maskedLImg


# ///// Make season-based composites /////
# Produces a list of imageCollections for each year. Each imageCollection contains the season-based composites for each year.
# Composites are created by taking the median of images in each group of the year.
def makeComposite(year, orgCollection):
    year = ee.Number(year)
    composite1 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(year, 9, 1), ee.Date.fromYMD(year, 12, 30)
        )
        .median()
        .set("system:time_start", ee.Date.fromYMD(year, 9, 1).millis())
        .set("Date", ee.Date.fromYMD(year, 9, 1))
    )

    composite2 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30),
        )
        .median()
        .set(
            "system:time_start",
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis(),
        )
        .set("Date", ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1))
    )

    composite3 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 8, 30),
        )
        .median()
        .set(
            "system:time_start",
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1).millis(),
        )
        .set("Date", ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1))
    )

    composite4 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 12, 30),
        )
        .median()
        .set(
            "system:time_start",
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1).millis(),
        )
        .set("Date", ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1))
    )

    # Return a collection of composites for the specific year
    return (
        ee.ImageCollection(composite1)
        .merge(ee.ImageCollection(composite2))
        .merge(ee.ImageCollection(composite3))
        .merge(ee.ImageCollection(composite4))
    )


# ///// Add day of year (DOY) to each image as a band /////
def addDOY(img):
    doy = img.date().getRelative("day", "year")
    doyBand = ee.Image.constant(doy).uint16().rename("doy")
    doyBand
    return img.addBands(doyBand)


# ///// Make metric-based imageCollections /////
# This groups images in a year and returns a list of imageCollections.


def groupImages(year, orgCollection, geometry):
    # This groups images and rename bands
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
    ]
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
    ]
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
    ]
    new_bandS2 = [
        "B_S2",
        "G_S2",
        "R_S2",
        "NIR_S2",
        "SWIR1_S2",
        "SWIR2_S2",
        "evi_S2",
        "gcvi_S2",
        "ndvi_S2",
        "sndvi_S2",
        "ndti_S2",
        "ndi5_S2",
        "ndi7_S2",
        "crc_S2",
        "sti_S2",
        "doy_S2",
    ]
    new_bandS3 = [
        "B_S3",
        "G_S3",
        "R_S3",
        "NIR_S3",
        "SWIR1_S3",
        "SWIR2_S3",
        "evi_S3",
        "gcvi_S3",
        "ndvi_S3",
        "sndvi_S3",
        "ndti_S3",
        "ndi5_S3",
        "ndi7_S3",
        "crc_S3",
        "sti_S3",
        "doy_S3",
    ]

    year = ee.Number(year)
    collection_1 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(year, 9, 1), ee.Date.fromYMD(year, 12, 30)
        )
        .filterBounds(geometry)
        .map(addDOY)
        .map(lambda img: img.select(bands).rename(new_bandS0))
    )

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(year, 9, 1).millis()))

    collection_2 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30),
        )
        .filterBounds(geometry)
        .map(addDOY)
        .map(lambda img: img.select(bands).rename(new_bandS1))
    )

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis()))

    collection_3 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 8, 30),
        )
        .filterBounds(geometry)
        .map(addDOY)
        .map(lambda img: img.select(bands).rename(new_bandS2))
    )

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1).millis()))

    collection_4 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 12, 30),
        )
        .filterBounds(geometry)
        .map(addDOY)
        .map(lambda img: img.select(bands).rename(new_bandS3))
    )

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1).millis()))

    # Return a list of imageCollections

    return [collection_1, collection_2, collection_3, collection_4]


# ///// Rename the bands of each composite in the imageCollections associated with each year /////


def renameComposites(collectionList):
    renamedCollectionList = []
    for i in range(len(collectionList)):
        ith_Collection = collectionList[i]
        Comp_S0 = ith_Collection.toList(ith_Collection.size()).get(0)
        Comp_S1 = ith_Collection.toList(ith_Collection.size()).get(1)
        Comp_S2 = ith_Collection.toList(ith_Collection.size()).get(2)
        Comp_S3 = ith_Collection.toList(ith_Collection.size()).get(3)

        bandsNot_to_rename = ["elevation", "slope", "aspect"]
        bands_to_rename = [
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
        ]
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
        ]
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
        ]
        new_bandS2 = [
            "B_S2",
            "G_S2",
            "R_S2",
            "NIR_S2",
            "SWIR1_S2",
            "SWIR2_S2",
            "evi_S2",
            "gcvi_S2",
            "ndvi_S2",
            "sndvi_S2",
            "ndti_S2",
            "ndi5_S2",
            "ndi7_S2",
            "crc_S2",
            "sti_S2",
        ]
        new_bandS3 = [
            "B_S3",
            "G_S3",
            "R_S3",
            "NIR_S3",
            "SWIR1_S3",
            "SWIR2_S3",
            "evi_S3",
            "gcvi_S3",
            "ndvi_S3",
            "sndvi_S3",
            "ndti_S3",
            "ndi5_S3",
            "ndi7_S3",
            "crc_S3",
            "sti_S3",
        ]

        composite_S0_renamed = (
            ee.Image(Comp_S0).select(bands_to_rename).rename(new_bandS0)
        )
        composite_S1_renamed = (
            ee.Image(Comp_S1).select(bands_to_rename).rename(new_bandS1)
        )
        composite_S2_renamed = (
            ee.Image(Comp_S2).select(bands_to_rename).rename(new_bandS2)
        )
        composite_S3_renamed = (
            ee.Image(Comp_S3).select(bands_to_rename).rename(new_bandS3)
        )

        composite_S0_Notrenamed = ee.Image(Comp_S0).select(bandsNot_to_rename)

        composite_S0 = ee.Image.cat([composite_S0_renamed, composite_S0_Notrenamed])

        renamedCollection = ee.ImageCollection.fromImages(
            [
                composite_S0,
                composite_S1_renamed,
                composite_S2_renamed,
                composite_S3_renamed,
            ]
        )
        renamedCollectionList = renamedCollectionList + [renamedCollection]
    return renamedCollectionList


# ///// Convert GEE list (ee.list) to python list /////
def eeList_to_pyList(eeList):
    pyList = []
    for i in range(eeList.size().getInfo()):
        pyList = pyList + [eeList.get(i)]
    return pyList


# ///// Convert python list to GEE list (ee.list)/////
def pyList_to_eeList(pyList):
    eeList = ee.List([])
    for i in range(len(pyList)):
        eeList = eeList.add(pyList[i])
    return eeList


# ///// Function to reduce each image in a collection (with different band
# names for each image) to
# a median value (median value over each field geometry) /////
def collectionReducer(imgcollection, shp):
    imageList = eeList_to_pyList(imgcollection.toList(imgcollection.size()))
    return list(
        map(
            lambda img: ee.Image(img).reduceRegions(
                **{"collection": shp, "reducer": ee.Reducer.median(), "scale": 1000}
            ),
            imageList,
        )
    )


# ///// Function to reduce each percentile image (with different band names for each image) to
# a median value (median value over each field geometry) /////
def percentile_imageReducer(imageList, shp):
    return list(
        map(
            lambda img: ee.Image(img).reduceRegions(
                **{
                    "reducer": ee.Reducer.median(),
                    "collection": shp,
                    "scale": 1000,
                    "tileScale": 16,
                }
            ),
            imageList,
        )
    )


# ///// Function to create pandas dataframes from geographically (by field) reduced featureCollections  /////
# Arguments: 1) List of lists of featureCollections:
#                              [[y1_f0, y1_f1, y1_f2, y1_f3], [y2_f0, y2_f1, y2_f2, y2_f3], ..., [yn_f0, yn_f1, yn_f2, yn_f3]]
#                              y1_f0 : season 1 (or time period 1) of year 1 reduced composite
# Output: Lists of dataframes. Each dataframe is the derived data for each year.
def eefeatureColl_to_Pandas(yearlyList, bandNames, important_columns_names):
    dataList = []  # This list is going to contain dataframes for each year data
    for i in range(len(yearlyList)):
        year_i = pyList_to_eeList(yearlyList[i])
        important_columns = important_columns_names + bandNames
        important_columns = [col for col in important_columns if col != "PolygonID"]
        important_columns += ["pointID"]

        df_yi = pd.DataFrame([])
        for j in range(year_i.length().getInfo()):
            f_j = year_i.get(j)  # Jth featureCollection (reduced composite data)
            df_j = eefeaturecoll_to_pandas_manual(
                ee.FeatureCollection(f_j)
            )  # Convert featureCollection to pandas dataframe
            df_j = df_j.rename(columns={"PolygonID": "pointID"})
            df_j = df_j[
                df_j.columns[(df_j.columns).isin(important_columns)]
            ]  # Pick needed columns
            df_yi = pd.concat([df_yi, df_j], axis=1)
        df_yi = df_yi.loc[
            :, ~df_yi.columns.duplicated()
        ]  # Drop repeated 'pointID' columns

        # reorder columns
        important_columns = [col for col in important_columns if col != "PolygonID"]
        important_columns += ["pointID"]
        df_yi = df_yi[important_columns]

        # Move pointID column to first position
        pointIDColumn = df_yi.pop("pointID")
        df_yi.insert(0, "pointID", pointIDColumn.iloc[:, 0])
        dataList = dataList + [df_yi]
    return dataList


# ///// Function to extract Gray-level Co-occurrence Matrix (GLCM) for each band in the composites  /////
# Input: an imageCollection containing the composites made for a year
# Output: List of imageCollections with GLCM bands.
def applyGLCM(coll):
    # Cast image values to a signed 32-bit integer.
    int32Coll = coll.map(lambda img: img.toInt32())
    glcmColl = int32Coll.map(
        lambda img: img.glcmTexture().set("system:time_start", img.date())
    )
    return glcmColl


## function to merge cdl wiht main data
def merge_cdl(maindf_list, cdl_list):
    """
    Merge main dataframe and cdl

    """
    cdl_list = [df.drop(columns="PriorCropT") for df in cdl_list]

    # Rename most_frequent_class to PriorCropT
    cdl_list = [
        df.rename(columns={"most_frequent_class": "ResidueType"}) for df in cdl_list
    ]

    # Select just priorCropT and pointID
    cdl_list = [df[["pointID", "ResidueType"]].copy() for df in cdl_list]

    # Rename cdl labels or crop type
    replacement_dict = {
        "24": "grain",
        "23": "grain",
        "51": "legume",
        "51": "legume",
        "31": "canola",
        "53": "legume",
        "21": "grain",
        "51": "legume",
        "52": "legume",
        "28": "grain",
    }

    # Just rename the three categories for grain, canola and legume
    cdl_list = [df.replace({"ResidueType": replacement_dict}) for df in cdl_list]
    cdl_list = [
        df.loc[df["ResidueType"].isin(["grain", "legume", "canola"])] for df in cdl_list
    ]

    # Make a list of tuples.Each tupple contain one seasonbased
    # Corresponding to a year and one cdl dataframe
    sat_cdl = list(zip(maindf_list, cdl_list))

    # Merge cdl with main dataframe
    final_dataframes_list = list(
        map(lambda df_tuple: pd.merge(df_tuple[0], df_tuple[1], on="pointID"), sat_cdl)
    )

    # move ResidueType to the 4th column
    [
        df.insert(3, "ResidueType", df.pop(df.columns[-1]))
        for df in final_dataframes_list
    ]

    return final_dataframes_list


def eefeaturecoll_to_pandas_manual(fc):
    features = fc.getInfo()["features"]
    dict_list = []
    for f in features:
        attr = f["properties"]
        dict_list.append(attr)
    df = pd.DataFrame(dict_list)
    return df


def renameComposites_S1(collectionList):
    renamedCollectionList = []
    for i in range(len(collectionList)):
        ith_Collection = collectionList[i]
        Comp_S0 = ith_Collection.toList(ith_Collection.size()).get(0)
        Comp_S1 = ith_Collection.toList(ith_Collection.size()).get(1)
        Comp_S2 = ith_Collection.toList(ith_Collection.size()).get(2)
        Comp_S3 = ith_Collection.toList(ith_Collection.size()).get(3)

        bands_to_rename = ["VV_dB", "VH_dB"]
        new_bandS0 = ["VV_S0", "VH_S0"]
        new_bandS1 = ["VV_S1", "VH_S1"]
        new_bandS2 = ["VV_S2", "VH_S2"]
        new_bandS3 = ["VV_S3", "VH_S3"]

        composite_S0_renamed = (
            ee.Image(Comp_S0).select(bands_to_rename).rename(new_bandS0)
        )
        composite_S1_renamed = (
            ee.Image(Comp_S1).select(bands_to_rename).rename(new_bandS1)
        )
        composite_S2_renamed = (
            ee.Image(Comp_S2).select(bands_to_rename).rename(new_bandS2)
        )
        composite_S3_renamed = (
            ee.Image(Comp_S3).select(bands_to_rename).rename(new_bandS3)
        )

        renamedCollection = ee.ImageCollection.fromImages(
            [
                composite_S0_renamed,
                composite_S1_renamed,
                composite_S2_renamed,
                composite_S3_renamed,
            ]
        )
        renamedCollectionList = renamedCollectionList + [renamedCollection]
    return renamedCollectionList


def eefeatureColl_to_Pandas_S1(yearlyList, bandNames, important_columns_names):
    dataList = []  # This list is going to contain dataframes for each year data
    for i in range(len(yearlyList)):
        year_i = pyList_to_eeList(yearlyList[i])
        important_columns = important_columns_names + bandNames

        df_yi = pd.DataFrame([])
        for j in range(year_i.length().getInfo()):
            f_j = year_i.get(j)  # Jth featureCollection (reduced composite data)
            # Convert featureCollection to pandas dataframe
            df_j = eefeaturecoll_to_pandas_manual(ee.FeatureCollection(f_j))

            df_j = df_j[
                df_j.columns[(df_j.columns).isin(important_columns)]
            ]  # Pick needed columns
            df_yi = pd.concat([df_yi, df_j], axis=1)
        # Drop repeated 'pointID' columns
        df_yi = df_yi.loc[:, ~df_yi.columns.duplicated()]

        # reorder columns
        df_yi = df_yi[important_columns]

        # Move pointID column to first position
        pointIDColumn = df_yi.pop("pointID")
        df_yi.insert(0, "pointID", pointIDColumn.iloc[:, 0])
        dataList = dataList + [df_yi]
    return dataList


def groupImages_S1(year, orgCollection, geometry):
    # This groups images and rename bands
    bands = ["VV_dB", "VH_dB", "doy"]
    new_bandS0 = ["VV_S0", "VH_S0", "doy_S0"]
    new_bandS1 = ["VV_S1", "VH_S1", "doy_S1"]
    new_bandS2 = ["VV_S2", "VH_S2", "doy_S2"]
    new_bandS3 = ["VV_S3", "VH_S3", "doy_S3"]

    year = ee.Number(year)
    collection_1 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(year, 9, 1), ee.Date.fromYMD(year, 12, 30)
        )
        .filterBounds(geometry)
        .map(addDOY)
        .map(lambda img: img.select(bands).rename(new_bandS0))
    )

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(year, 9, 1).millis()))

    collection_2 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30),
        )
        .filterBounds(geometry)
        .map(addDOY)
        .map(lambda img: img.select(bands).rename(new_bandS1))
    )

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis()))

    collection_3 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 8, 30),
        )
        .filterBounds(geometry)
        .map(addDOY)
        .map(lambda img: img.select(bands).rename(new_bandS2))
    )

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1).millis()))

    collection_4 = (
        orgCollection.filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 12, 30),
        )
        .filterBounds(geometry)
        .map(addDOY)
        .map(lambda img: img.select(bands).rename(new_bandS3))
    )

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1).millis()))

    # Return a list of imageCollections

    return [collection_1, collection_2, collection_3, collection_4]


# function to convert geopandas dataframe to feature collection
def geopandas_to_ee(df):
    gdf = gpd.GeoDataFrame(df, geometry=df["geometry"])

    # Convert the GeoDataFrame to a GeoJSON string
    geojson_str = gdf.to_json()

    # Parse the GeoJSON string to a dictionary
    geojson_dict = json.loads(geojson_str)

    # Create Earth Engine Features from GeoJSON features
    features = [ee.Feature(feat) for feat in geojson_dict["features"]]

    # Create a FeatureCollection from the features
    feature_collection = ee.FeatureCollection(features)
    return feature_collection


# -

# #### Imports

# +
######## imports #########
# consider a polygon that covers the study area (Whitman & Columbia counties)
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
)

geometry2 = (ee.Geometry.Point([-117.10053796709163, 46.94957951590986]),)

asset_folder = "projects/ee-bio-ag-tillage/assets/tillmap_test_2017"
assets_list = ee.data.getList({"id": asset_folder})
shpfilesList_ = [i["id"] for i in assets_list]

path_to_data = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
    "Projects/Tillage_Mapping/Data/"
)

# path_to_data = ('/home/amnnrz/OneDrive - a.norouzikandelati/Ph.D/Projects/'
#                 'Tillage_Mapping/Data/')

startYear = 2011
endYear = 2012

# +
path_to_data = ("/Users/aminnorouzi/Library/CloudStorage/"
                "OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/"
                "Projects/Tillage_Mapping/Data/GIS_Data/2012_2017_wsda/")

gpd_2012 = gpd.read_file(path_to_data + "WSDA_2012.shp")
gpd_2017 = gpd.read_file(path_to_data + "WSDA_2017.shp")

eastern_counties = [
    "Whitman",
    "Columbia",
    "Adams",
    "Garfield",
    "Asotin",
    "Lincoln",
    "Douglas",
    "Grant",
    "Benton",
    "Franklin",
    "Spokane",
    "Wallawala"
]
gpd_df_2017 = gpd_2017.loc[gpd_2017["County"].isin(eastern_counties)]
gpd_df_2017["CropType"].unique()
gpd_df_2012 = gpd_2012.loc[gpd_2012["County"].isin(eastern_counties)]
gpd_df_2012["CropType"].unique()


selected_crops = [
    "Wheat",
    "Wheat Fallow",
    "Pea, Green",
    "Rye",
    "Barley",
    "Chickpea",
    "Pea, Dry",
    "Barley Hay",
    "Canola",
    "Triticale",
    "Bean, Dry",
    "Oat",
    "Pea Seed",
    "Oat Hay",
    "Sorghum",
    "Buckwheat",
    "Lentil",
    "Triticale Hay",
    "Cereal Grain, Unknown",
    "Legume Cover",
]

gpd_df_2017_filtered = gpd_df_2017.loc[gpd_df_2017["CropType"].isin(selected_crops)]
gpd_df_2012_filtered = gpd_df_2012.loc[gpd_df_2012["CropType"].isin(selected_crops)]
gpd_df_filtered.groupby(["County"])["ExactAcres"].sum()

gpd_df_filtered = gpd_df_2017_filtered

# +
All_crops = ['Apple', 'Sugar Beet', 'Wheat', 'Fallow', 'Grape, Wine',
       'Corn, Field', 'Grape, Juice', 'Onion', 'Potato', 'Cherry', 'Wheat Fallow', 'Corn, Sweet',
       'Pea, Green', 'Alfalfa Hay', 'Grass Hay', 'Hops', 'Fallow, Idle', 'Market Crops', 'Blueberry',
       'Bluegrass Seed', 'Carrot', 'Mint', 'Apricot', 'Yellow Mustard', 'Driving Range', 'Buckwheat', 'Walnut',
       'Nursery, Ornamental', 'Alfalfa/Grass Hay', 'Pea Seed', 'Timothy',
       'Sorghum', 'Poplar', 'Nursery, Orchard/Vineyard', 'Pear', 'Caneberry',
       'Grape, Table', 'Marijuana', 'Asparagus', 'Triticale',
       'Nectarine/Peach', 'Plum', 'Bean Seed', 'Sudangrass', 'Watermelon',
       'Oat Hay', 'Triticale Hay', 'Strawberry', 'Pumpkin', 'Bean, Green',
       'Barley', 'Bean, Dry', 'Clover/Grass Hay', 'Silviculture', 'Rye',
       'Kale', 'Grape, Unknown', 'Medicinal Herb', 'Fallow, Tilled',
       'Sunflower', 'Sugar Beet Seed', 'Canola', 'Corn Seed',
       'Onion Seed', 'Cilantro Seed', 'Sod Farm',
       'Sunflower Seed', 'Pea, Dry', 'Barley Hay', 'Alfalfa Seed',
       'Filbert', 'Squash', 'Grass Seed, Other', 'Chickpea',
       'Radish Seed', 'Alkali Bee Bed', 'Carrot Seed', 'Pepper',
       'Vegetable, Unknown', 'Mustard', 'Nursery, Lavender', 'Beet Seed',
       'Spinach', 'Oat', 'Potato Seed', 'Green Manure', 'Leek',
       'Ryegrass Seed', 'Corn, Unknown', 'Melon, Unknown', 'Fescue Seed',
       'Legume Cover', 'Flax Seed', 'Chestnut', 'Cabbage',
       'Reclamation Seed', 'Nursery, Silviculture', 'Soybean',
       'Clover Seed', 'Yarrow Seed', 'Christmas Tree', 'Dill', 'Rosemary',
       'Bromegrass Seed', 'Cantaloupe', 'Lentil', 'Flax',
       'Cereal Grain, Unknown', 'Rye Hay', 'Cranberry']

gpd_df_2017_all_crops = gpd_df_2017.loc[gpd_df_2017["CropType"].isin(All_crops)]
gpd_df_2012_all_crops = gpd_df_2012.loc[gpd_df_2012["CropType"].isin(All_crops)]
# -

gpd_df_2017_filtered["Acres"].sum() / gpd_df_2017_all_crops["Acres"].sum()

gpd_df_2017['CropType'].unique()

# +
gpd_df_2017_all_crops.groupby("CropType")["Acres"].sum()

# Group by 'cropType' and sum 'acres'
grouped = gpd_df_2017_all_crops.groupby("CropType")["Acres"].sum()

# Calculate the total acres
total_acres = grouped.sum()

# Calculate the percentage of acres for each crop type
percentage = (grouped / total_acres) * 100
# -

percentage

# +
# Split shapefile into shapefiles with smaller numbers of polygons
import os
import geopandas as gpd
from math import ceil

path_to_large_shapefiles = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/field_level_data/"
    "mapping_data/"
)

path_to_batches = path_to_large_shapefiles + "2017/batches"


def split_shapefile(input_shapefile, output_directory, max_features_per_file):
    # Read the original shapefile
    gdf = gpd.read_file(input_shapefile)
    total_features = len(gdf)
    number_of_splits = ceil(total_features / max_features_per_file)

    # Split and save shapefiles
    for i in range(number_of_splits):
        start = i * max_features_per_file
        end = start + max_features_per_file
        split_gdf = gdf.iloc[start:end]
        split_filename = os.path.join(output_directory, f"split_{i}.shp")
        split_gdf.to_file(split_filename)

    print(f"Shapefile split into {number_of_splits} parts.")


shpfile_2022_path = path_to_large_shapefiles + "2017/2017shpfile.shp"
split_shapefile(shpfile_2022_path, path_to_batches, 500)
# -

# # Download CDL data

# +
# Load the USDA NASS CDL dataset
cdl = (
    ee.ImageCollection("USDA/NASS/CDL")
    .filterDate(
        ee.Date.fromYMD(ee.Number(startYear), 1, 1),
        ee.Date.fromYMD(ee.Number(endYear), 12, 31),
    )
    .filterBounds(geometry)
)

def getMostFrequentClass(image, polygon):
    polygon = ee.Feature(polygon)
    # Clip the CDL image to the polygon
    cdlClipped = image.clip(polygon)

    # Calculate the histogram for the polygon
    histogram = cdlClipped.reduceRegion(
        **{
            "reducer": ee.Reducer.frequencyHistogram(),
            "geometry": polygon.geometry(),
            "scale": 30,  # adjust based on CDL dataset's resolution
        }
    ).get("cropland")

    # Convert the dictionary to key-value pairs
    keys = ee.Dictionary(histogram).keys()
    values = keys.map(lambda key: ee.Number(ee.Dictionary(histogram).get(key)))

    # Find the most frequent class
    max = values.reduce(ee.Reducer.max())
    indexMax = values.indexOf(max)
    mostFrequentClass = keys.get(indexMax)

    return polygon.set("most_frequent_class", mostFrequentClass)


# filter collection for the year
years = np.arange(startYear, endYear)


def cdl_dataframe_yith(year, gpd_split_ith):
    cdl_image = cdl.filterDate(
        ee.Date.fromYMD(ee.Number(int(year)), 1, 1),
        ee.Date.fromYMD(ee.Number(int(year)), 12, 31),
    ).first()

    # Map the function across all polygons
    polygons = geemap.geopandas_to_ee(gpd_split_ith)
    polygonsList = eeList_to_pyList(polygons.toList(polygons.size()))

    cdl_feature_pylist = list(
        map(lambda pol: getMostFrequentClass(cdl_image, pol), polygonsList)
    )

    cdl_feature_eelist = pyList_to_eeList(cdl_feature_pylist)

    cdl_y_data = ee.FeatureCollection(cdl_feature_eelist)
    return cdl_y_data


# +
import geemap
import numpy as np
import cProfile
import pstats

year = 2016

# Assuming the split_shapefile function has already been run
split_shapefiles = [f for f in os.listdir(path_to_batches) if f.endswith(".shp")]


def submit_batch_task(shapefile_name, year):
    shapefile_path = os.path.join(path_to_batches, shapefile_name)
    shapefile_gdf = gpd.read_file(shapefile_path)
    cdl_y_df = cdl_dataframe_yith(year, shapefile_gdf)

    task = ee.batch.Export.table.toDrive(
        collection=cdl_y_df,
        description=shapefile_name,
        folder="CDL_" + str(year),
        fileFormat="CSV",
    )
    task.start()
    return task


# Sorting the list
# The key is to extract the numeric part and convert it to integer for correct sorting
sorted_file_list = sorted(
    split_shapefiles, key=lambda x: int(x.split("_")[1].split(".")[0])
)



tasks = []
for year in np.arange(startYear, endYear):
    for shapefile_name in sorted_file_list[18:]:
        print(shapefile_name)
        task = submit_batch_task(shapefile_name, year)
        tasks.append(task)


# # Now profile the main_function
# cProfile.run("main_func()", "profile_stats")

# # Read and print out the stats
# p = pstats.Stats("profile_stats")
# p.sort_stats("cumulative").print_stats(
#     10
# )  # Adjust the number to view more or fewer lines
# -

def check_task_status(tasks):
    for task in tasks:
        print(task.status())


# + [markdown] id="Xi8j9i9nSiW7"
# # Download Season-Based Landsat Data

# +
#####################################################################
###################      Season-based Features      #################
#####################################################################
# Extract season-based features, using main bands, Indices and
# their Gray-level Co-occurence Metrics (GLCMs)
# import USGS Landsat 8 Level 2, Collection 2, Tier 1

# path_to_large_shapefiles = (
#     "/Users/aminnorouzi/Library/CloudStorage/"
#     "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
#     "Ph.D/Projects/Tillage_Mapping/Data/field_level_data/"
#     "mapping_data/2012/"
# )
path_to_large_shapefiles = (
    "/home/amnnrz/OneDrive - a.norouzikandelati/"
    "Ph.D/Projects/Tillage_Mapping/Data/field_level_data/"
    "mapping_data/2017/"
)

path_to_batches = path_to_large_shapefiles + "batches"

split_shapefiles = [f for f in os.listdir(path_to_batches) if f.endswith(".shp")]

sorted_file_list = sorted(
    split_shapefiles, key=lambda x: int(x.split("_")[1].split(".")[0])
)

pointID_start = 0
geopandas_list = []
for shp_name in sorted_file_list:
    shapefile_path = os.path.join(path_to_batches, shp_name)
    shapefile_gdf = gpd.read_file(shapefile_path)
    shapefile_gdf["PolygonID"] = pd.Series(
        np.arange(pointID_start, pointID_start + shapefile_gdf.shape[0])
    )
    pointID_start += shapefile_gdf.shape[0]
    geopandas_list.append(shapefile_gdf)

# gpd_list = []
# for df in geopandas_list:
#     df = df.drop(columns=["Organic"])
#     gpd_list.append(df)
geopandas_list = [geemap.geopandas_to_ee(i) for i in geopandas_list]

startYear = 2017
endYear = 2018

L8T1 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
L7T1 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")

L8_2122 = (
    L8T1.filter(ee.Filter.calendarRange(startYear, endYear, "year"))
    .map(lambda img: img.set("year", img.date().get("year")))
    .map(lambda img: img.clip(geometry))
)

L7_2122 = (
    L7T1.filter(ee.Filter.calendarRange(startYear, endYear, "year"))
    .map(lambda img: img.set("year", img.date().get("year")))
    .map(lambda img: img.clip(geometry))
)

# Apply scaling factor
L8_2122 = L8_2122.map(applyScaleFactors)
L7_2122 = L7_2122.map(applyScaleFactors)

# Rename bands
L8_2122 = L8_2122.map(renameBandsL8)
L7_2122 = L7_2122.map(renameBandsL7)

# Merge Landsat 7 and 8 collections
landSat_7_8 = ee.ImageCollection(L8_2122.merge(L7_2122))

# Apply NDVI mask
landSat_7_8 = landSat_7_8.map(addNDVI)

landSat_7_8 = landSat_7_8.map(lambda image: maskNDVI(image, threshold=0.3))

# Mask Clouds
landSat_7_8 = landSat_7_8.map(cloudMaskL8)

# Mask prercipitation > 3mm two days prior
# import Gridmet collection
GridMet = (
    ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
    .filter(
        ee.Filter.date(
            ee.Date.fromYMD(ee.Number(startYear), 1, 1),
            ee.Date.fromYMD(ee.Number(endYear), 12, 30),
        )
    )
    .filterBounds(geometry)
)
landSat_7_8 = landSat_7_8.map(lambda image: MoistMask(image, GridMet))

# Add spectral indices to each in the collection as bands
landSat_7_8 = landSat_7_8.map(addIndices)

# Add Terrain variables (elevation, slope and aspect)
dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
slope = ee.Terrain.slope(dem)
aspect = ee.Terrain.aspect(dem)

# Merge Terrain variables with landsat image collection
landSat_7_8 = (
    landSat_7_8.map(lambda image: image.addBands(dem))
    .map(lambda image: image.addBands(slope))
    .map(lambda image: image.addBands(aspect))
)

# Create season-based composites
# Specify time period

years = list(range(startYear, endYear))

# Create season-based composites for each year and put them in a list
yearlyCollectionsList = []
for y in years:
    yearlyCollectionsList = yearlyCollectionsList + [makeComposite(y, landSat_7_8)]
# Rename bands of each composite in each yearly collection
renamedCollectionList = renameComposites(yearlyCollectionsList)

# Select shapefile
# WSDA_featureCol = ee.FeatureCollection(
#   'projects/ee-bio-ag-tillage/assets/' + f'{shp}')
shpfilesList_ = geopandas_list

# def reproject_feature(feature):
#     # Reproject the geometry of the feature
#     reprojected_geometry = feature.geometry().transform("EPSG:4326", 0.001)
#     # Return the feature with the reprojected geometry
#     return ee.Feature(reprojected_geometry, feature.toDictionary())

# shpfilesList_ = [fc.map(reproject_feature) for fc in shpfilesList_]


df_seasonBased_list_2012 = []
for idx, shp_batch in enumerate(shpfilesList_):
    print(idx)
    shpfilesList = [shp_batch]
    # Clip each collection to the WSDA field boundaries
    clipped_mainBands_CollectionList = list(
        map(
            lambda collection, shp: collection.map(
                lambda img: img.clip(ee.FeatureCollection(shp))
            ),
            renamedCollectionList,
            shpfilesList,
        )
    )

    # Extract GLCM metrics
    clipped_GLCM_collectionList = list(map(applyGLCM, clipped_mainBands_CollectionList))

    # Extract band names to use in our dataframes
    # The bands are identical for all years so we use the first year imageCollection, [0]
    # Main bands:
    imageList = eeList_to_pyList(
        clipped_mainBands_CollectionList[0].toList(
            clipped_mainBands_CollectionList[0].size()
        )
    )
    nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(), imageList))
    mainBands = [name for sublist in nameLists for name in sublist]

    # GLCM bands:
    imageList = eeList_to_pyList(
        clipped_GLCM_collectionList[0].toList(clipped_GLCM_collectionList[0].size())
    )
    nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(), imageList))
    glcmBands = [name for sublist in nameLists for name in sublist]

    # Reduce each image in the imageCollections (with main bands) to
    # mean value over each field (for each year). This will produce a list of
    # lists containing reduced featureCollections
    reducedList_mainBands = list(
        map(
            lambda collection, shp: collectionReducer(
                collection, ee.FeatureCollection(shp)
            ),
            clipped_mainBands_CollectionList,
            shpfilesList,
        )
    )

    # Reduce each image in the imageCollections (with GLCM bands) to mean value
    # over each field (for each year)
    reducedList_glcmBands = list(
        map(
            lambda collection, shp: collectionReducer(
                collection, ee.FeatureCollection(shp)
            ),
            clipped_GLCM_collectionList,
            shpfilesList,
        )
    )

    # Convert each year's composites to a single dataframe and put all
    # the dataframes in a list
    # important_columns_names = [
    #     "pointID",
    #     "CurrentCro",
    #     "DateTime",
    #     "PriorCropT",
    #     "ResidueCov",
    #     "Tillage",
    #     "WhereInRan",
    # ]

    fc = ee.FeatureCollection(shpfilesList[0])
    important_columns_names = fc.first().propertyNames().getInfo()
    important_columns_names = [
        string for string in important_columns_names if string != "system:index"
    ]

    # important_columns_names = list(
    #     geemap.ee_to_pandas(ee.FeatureCollection(shpfilesList[0])).columns.values
    # )

    seasonBased_dataframeList_mainBands = eefeatureColl_to_Pandas(
        reducedList_mainBands, mainBands, important_columns_names
    )
    seasonBased_dataframeList_glcm = eefeatureColl_to_Pandas(
        reducedList_glcmBands, glcmBands, important_columns_names
    )

    # Merge main and glcm bands for each year
    allYears_seasonBased_list = list(
        map(
            lambda mainband_df, glcmband_df: pd.concat(
                [mainband_df, glcmband_df], axis=1
            ),
            seasonBased_dataframeList_mainBands,
            seasonBased_dataframeList_glcm,
        )
    )

    # Remove duplicated columns
    duplicated_cols_idx = [df.columns.duplicated() for df in allYears_seasonBased_list]
    Landsat_seasonBased_list = list(
        map(
            lambda df, dup_idx: df.iloc[:, ~dup_idx],
            allYears_seasonBased_list,
            duplicated_cols_idx,
        )
    )

    print(Landsat_seasonBased_list[0].shape)
    Landsat_seasonBased_list[0].to_csv(
        path_to_large_shapefiles + "seasonBased_batches_csv/" + f"batch_{idx}.csv"
    )
    df_seasonBased_list_2012.append(Landsat_seasonBased_list)
    # Display on Map
    # Map = geemap.Map()
    # Map.setCenter(-117.100, 46.94, 7)
    # Map.addLayer(ee.Image(clipped_mainBands_CollectionList[0].toList(
    # clipped_mainBands_CollectionList[0].size(
    # )).get(1)), {'bands': ['B4_S1', 'B3_S1', 'B2_S1'], max: 0.5, 'gamma': 2}, 'L8')
    # Map
# -

sorted_file_list

geopandas_list

df = pd.concat([_[0] for _ in df_seasonBased_list_2012])
df.to_csv(path_to_large_shapefiles + f"seasonBased_all.csv")

# + [markdown] id="DUhdHR8xIrUE"
# # Download Metric-Based Landsat Data for training 

# + colab={"background_save": true} id="vrRY7E6NLhul"
from functools import reduce

###########################################################################
###################      Distribution-based Features      #################
###########################################################################
#### Extract distribution-based (metric-based) features using main bands,
#### indices and Gray-level Co-occurence Metrics (GLCMs)

# Create metric composites
# Years
years = list(range(startYear, endYear))

# Create a list of lists of imageCollections. Each year would have n number
# of imageCollection corresponding to the time periods specified
# for creating metric composites.
yearlyCollectionsList = []
for y in years:
    yearlyCollectionsList = yearlyCollectionsList + [
        groupImages(y, landSat_7_8, geometry)
    ]  # 'yearlyCollectionsList' is a Python list
# yearlyCollectionsList[0][0]

df_metricBased_list_2017 = []
for shp_batch in shpfilesList_:
    shpfilesList = [shp_batch]
    # Clip each collection to the WSDA field boundaries
    clipped_mainBands_CollectionList = list(
        map(
            lambda collList, shp: list(
                map(
                    lambda collection: ee.ImageCollection(collection).map(
                        lambda img: img.clip(ee.FeatureCollection(shp))
                    ),
                    collList,
                )
            ),
            yearlyCollectionsList,
            shpfilesList,
        )
    )

    # Extract GLCM metrics
    clipped_GLCM_collectionList = list(
        map(
            lambda collList: list(map(applyGLCM, collList)),
            clipped_mainBands_CollectionList,
        )
    )

    # # Compute percentiles
    percentiles = [5, 25, 50, 75, 100]
    mainBands_percentile_collectionList = list(
        map(
            lambda collList: list(
                map(
                    lambda collection: collection.reduce(
                        ee.Reducer.percentile(percentiles)
                    ),
                    collList,
                )
            ),
            clipped_mainBands_CollectionList,
        )
    )

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

    # Reduce each image in the imageCollections (with main bands) to mean
    #  value over each field (for each year)
    # This will produce a list of lists containing reduced featureCollections
    reducedList_mainBands = list(
        map(
            lambda imgList, shp: percentile_imageReducer(
                imgList, ee.FeatureCollection(shp)
            ),
            mainBands_percentile_collectionList,
            shpfilesList,
        )
    )

    # Reduce each image in the imageCollections (with GLCM bands)
    #  to mean value over each field (for each year)
    reducedList_glcmBands = list(
        map(
            lambda imgList, shp: percentile_imageReducer(
                imgList, ee.FeatureCollection(shp)
            ),
            glcmBands_percentile_collectionList,
            shpfilesList,
        )
    )

    # Extract band names to use in our dataframes
    # The bands are identical for all years so we use the first year
    #  imageCollection, [0]
    # Main bands:
    nameLists = list(
        map(
            lambda img: ee.Image(img).bandNames().getInfo(),
            mainBands_percentile_collectionList[0],
        )
    )
    mainBands = [name for sublist in nameLists for name in sublist]

    # GLCM bands:
    nameLists = list(
        map(
            lambda img: ee.Image(img).bandNames().getInfo(),
            glcmBands_percentile_collectionList[0],
        )
    )
    glcmBands = [name for sublist in nameLists for name in sublist]

    # Convert each year's composites to a single dataframe
    # and put all the dataframes in a list a dataframe.

    # important_columns_names = [
    #     "pointID",
    #     "CurrentCro",
    #     "DateTime",
    #     "PriorCropT",
    #     "ResidueCov",
    #     "Tillage",
    #     "WhereInRan",
    # ]

    important_columns_names = list(
        geemap.ee_to_pandas(ee.FeatureCollection(shpfilesList[0])).columns.values
    )
    
    metricBased_dataframeList_mainBands = eefeatureColl_to_Pandas(
        reducedList_mainBands, mainBands, important_columns_names
    )

    metricBased_dataframeList_glcm = eefeatureColl_to_Pandas(
        reducedList_glcmBands, glcmBands, important_columns_names
    )

    # Merge main and glcm bands for each year
    allYears_metricBased_list = list(
        map(
            lambda mainband_df, glcmband_df: pd.concat(
                [mainband_df, glcmband_df], axis=1
            ),
            metricBased_dataframeList_mainBands,
            metricBased_dataframeList_glcm,
        )
    )

    # Remove duplicated columns
    duplicated_cols_idx = [df.columns.duplicated() for df in allYears_metricBased_list]
    Landsat_metricBased_list = list(
        map(
            lambda df, dup_idx: df.iloc[:, ~dup_idx],
            allYears_metricBased_list,
            duplicated_cols_idx,
        )
    )

    print(Landsat_metricBased_list[0].shape)
    df_metricBased_list_2017.append(Landsat_metricBased_list)
# -

# # Download Metric-Based Landsat Data for test

# +
from functools import reduce

path_to_save_batches = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/field_level_data/"
    "mapping_data/"
)

# read shapefiles as geopandas dataframes and put them in a list
files = os.listdir(path_to_data + "GIS_Data/shapefiles_2017_map")

shapefile_names = [shp for shp in files if shp.endswith(".shp")]


geopandas_list = [
    gpd.read_file(path_to_data + "GIS_Data/shapefiles_2017_map/" + _)
    for _ in shapefile_names
]

geopandas_list = [geopandas_to_ee(i) for i in geopandas_list]

###########################################################################
###################      Distribution-based Features      #################
###########################################################################
#### Extract distribution-based (metric-based) features using main bands,
#### indices and Gray-level Co-occurence Metrics (GLCMs)

# Create metric composites
# Years
years = list(range(startYear, endYear))

# Create a list of lists of imageCollections. Each year would have n number
# of imageCollection corresponding to the time periods specified
# for creating metric composites.
yearlyCollectionsList = []
for y in years:
    yearlyCollectionsList = yearlyCollectionsList + [
        groupImages(y, landSat_7_8, geometry)
    ]  # 'yearlyCollectionsList' is a Python list
# yearlyCollectionsList[0][0]

df_metricBased_list_2017 = []
for i in np.arange(len(geopandas_list)):
    # Clip each collection to the WSDA field boundaries
    clipped_mainBands_CollectionList = list(
        map(
            lambda collList, shp: list(
                map(
                    lambda collection: ee.ImageCollection(collection).map(
                        lambda img: img.clip(shp)
                    ),
                    collList,
                )
            ),
            yearlyCollectionsList,
            [geopandas_list[i]],
        )
    )

    # Extract GLCM metrics
    clipped_GLCM_collectionList = list(
        map(
            lambda collList: list(map(applyGLCM, collList)),
            clipped_mainBands_CollectionList,
        )
    )

    # # Compute percentiles
    percentiles = [5, 25, 50, 75, 100]
    mainBands_percentile_collectionList = list(
        map(
            lambda collList: list(
                map(
                    lambda collection: collection.reduce(
                        ee.Reducer.percentile(percentiles)
                    ),
                    collList,
                )
            ),
            clipped_mainBands_CollectionList,
        )
    )

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

    # Reduce each image in the imageCollections (with main bands) to mean
    #  value over each field (for each year)
    # This will produce a list of lists containing reduced featureCollections
    reducedList_mainBands = list(
        map(
            lambda imgList, shp: percentile_imageReducer(imgList, shp),
            mainBands_percentile_collectionList,
            [geopandas_list[i]],
        )
    )

    # Reduce each image in the imageCollections (with GLCM bands)
    #  to mean value over each field (for each year)
    reducedList_glcmBands = list(
        map(
            lambda imgList, shp: percentile_imageReducer(imgList, shp),
            glcmBands_percentile_collectionList,
            [geopandas_list[i]],
        )
    )

    # Extract band names to use in our dataframes
    # The bands are identical for all years so we use the first year
    #  imageCollection, [0]
    # Main bands:
    nameLists = list(
        map(
            lambda img: ee.Image(img).bandNames().getInfo(),
            mainBands_percentile_collectionList[0],
        )
    )
    mainBands = [name for sublist in nameLists for name in sublist]

    # GLCM bands:
    nameLists = list(
        map(
            lambda img: ee.Image(img).bandNames().getInfo(),
            glcmBands_percentile_collectionList[0],
        )
    )
    glcmBands = [name for sublist in nameLists for name in sublist]

    # Convert each year's composites to a single dataframe
    # and put all the dataframes in a list a dataframe.

    # important_columns_names = [
    #     "pointID",
    #     "CurrentCro",
    #     "DateTime",
    #     "PriorCropT",
    #     "ResidueCov",
    #     "Tillage",
    #     "WhereInRan",
    # ]

    important_columns_names = list(
        geemap.ee_to_pandas(ee.FeatureCollection(shpfilesList[0])).columns.values
    )

    metricBased_dataframeList_mainBands = eefeatureColl_to_Pandas(
        reducedList_mainBands, mainBands, important_columns_names
    )

    metricBased_dataframeList_glcm = eefeatureColl_to_Pandas(
        reducedList_glcmBands, glcmBands, important_columns_names
    )

    # Merge main and glcm bands for each year
    allYears_metricBased_list = list(
        map(
            lambda mainband_df, glcmband_df: pd.concat(
                [mainband_df, glcmband_df], axis=1
            ),
            metricBased_dataframeList_mainBands,
            metricBased_dataframeList_glcm,
        )
    )

    # Remove duplicated columns
    duplicated_cols_idx = [df.columns.duplicated() for df in allYears_metricBased_list]
    Landsat_metricBased_list = list(
        map(
            lambda df, dup_idx: df.iloc[:, ~dup_idx],
            allYears_metricBased_list,
            duplicated_cols_idx,
        )
    )

    print(Landsat_metricBased_list[0].shape)
    df_metricBased_list_2017.append(Landsat_metricBased_list)
# -

# # Download Season-Based Sentinel-1 Data

# +
Sentinel_1 = (
    ee.ImageCollection("COPERNICUS/S1_GRD")
    .filter(ee.Filter.calendarRange(startYear, endYear, "year"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .map(lambda img: img.set("year", img.date().get("year")))
    .map(lambda img: img.clip(geometry))
)


# Convert pixel values to logarithmic scale (decible scale)
def toDb(img):
    dB = ee.Image(10.0).multiply(img.log10()).toFloat()
    # Rename the bands for VV and VH
    bands = img.bandNames()
    newBands = bands.map(lambda band: ee.String(band).cat("_dB"))

    # Add dB bands and rename them
    imageWithDb = img.addBands(dB)
    renamedImage = imageWithDb.select(bands, newBands)

    return renamedImage


# Apply preprocessing and visualization
processedCollection = Sentinel_1.map(toDb).map(
    lambda img: img.select(["VV_dB", "VH_dB"])
)

# # Display on map
# Map = geemap.Map(center=[46.94, -117.100], zoom=7)
# Map.addLayer(processedCollection, {
#              'bands': ['VV_dB', 'VH_dB'], 'min': -20, 'max': 0}, 'Sentinel-1')
# Map

# Specify time period
years = list(range(startYear, endYear))

yearlyCollectionsList = []
for y in years:
    yearlyCollectionsList = yearlyCollectionsList + [
        makeComposite(y, processedCollection)
    ]

renamedCollectionList = renameComposites_S1(yearlyCollectionsList)

df_seasonBased_list_2017_sentinel1 = []
for shp_batch in shpfilesList_:
    shpfilesList = [shp_batch]
    clipped_mainBands_CollectionList = list(
        map(
            lambda collection, shp: collection.map(
                lambda img: img.clip(ee.FeatureCollection(shp))
            ),
            renamedCollectionList,
            shpfilesList,
        )
    )

    clipped_mainBands_CollectionList

    clipped_GLCM_collectionList = list(map(applyGLCM, clipped_mainBands_CollectionList))

    clipped_GLCM_collectionList = list(map(applyGLCM, clipped_mainBands_CollectionList))

    imageList = eeList_to_pyList(
        clipped_mainBands_CollectionList[0].toList(
            clipped_mainBands_CollectionList[0].size()
        )
    )
    nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(), imageList))
    mainBands = [name for sublist in nameLists for name in sublist]

    # GLCM bands:
    imageList = eeList_to_pyList(
        clipped_GLCM_collectionList[0].toList(clipped_GLCM_collectionList[0].size())
    )
    nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(), imageList))
    glcmBands = [name for sublist in nameLists for name in sublist]

    # Reduce each image in the imageCollections (with main bands) to
    # mean value over each field (for each year). This will produce a list of
    # lists containing reduced featureCollections
    reducedList_mainBands = list(
        map(
            lambda collection, shp: collectionReducer(
                collection, ee.FeatureCollection(shp)
            ),
            clipped_mainBands_CollectionList,
            shpfilesList,
        )
    )

    # Reduce each image in the imageCollections (with GLCM bands) to mean value
    # over each field (for each year)
    reducedList_glcmBands = list(
        map(
            lambda collection, shp: collectionReducer(
                collection, ee.FeatureCollection(shp)
            ),
            clipped_GLCM_collectionList,
            shpfilesList,
        )
    )

    # Convert each year's composites to a single dataframe and put all
    # the dataframes in a list
    # important_columns_names = [
    #     "pointID",
    #     "CurrentCro",
    #     "DateTime",
    #     "PriorCropT",
    #     "ResidueCov",
    #     "Tillage",
    #     "WhereInRan",
    # ]
    important_columns_names = list(
        geemap.ee_to_pandas(ee.FeatureCollection(shpfilesList[0])).columns.values
    )
    seasonBased_dataframeList_mainBands = eefeatureColl_to_Pandas_S1(
        reducedList_mainBands, mainBands, important_columns_names
    )
    seasonBased_dataframeList_glcm = eefeatureColl_to_Pandas_S1(
        reducedList_glcmBands, glcmBands, important_columns_names
    )

    # Merge main and glcm bands for each year
    allYears_seasonBased_list = list(
        map(
            lambda mainband_df, glcmband_df: pd.concat(
                [mainband_df, glcmband_df], axis=1
            ),
            seasonBased_dataframeList_mainBands,
            seasonBased_dataframeList_glcm,
        )
    )

    # Remove duplicated columns
    duplicated_cols_idx = [df.columns.duplicated() for df in allYears_seasonBased_list]
    Sentinel_1_seasonBased_list = list(
        map(
            lambda df, dup_idx: df.iloc[:, ~dup_idx],
            allYears_seasonBased_list,
            duplicated_cols_idx,
        )
    )

    print(Sentinel_1_seasonBased_list[0].shape)
    df_seasonBased_list_2017_sentinel1.append(Sentinel_1_seasonBased_list)
# -

df = pd.concat([_[0] for _ in df_seasonBased_list_2017_sentinel1])
df.to_csv(path_to_save_batches + f"/2017/seasonBased_sentinel1_all.csv")

# # Download Metric-Based Sentinel-1 Data

# +
from functools import reduce
###########################################################################
###################      Distribution-based Features      #################
###########################################################################
# Create a list of lists of imageCollections. Each year would have n number 
# of imageCollection corresponding to the time periods specified
# for creating metric composites.
yearlyCollectionsList = []
for y in years:
  yearlyCollectionsList = yearlyCollectionsList + \
  [groupImages_S1(y, processedCollection, geometry)]  # 'yearlyCollectionsList' is a Python list

# Clip each collection to the WSDA field boundaries
clipped_mainBands_CollectionList = list(map(
  lambda collList, shp: list(map(
    lambda collection: ee.ImageCollection(collection).map(
      lambda img: img.clip(ee.FeatureCollection(shp))), collList)),
        yearlyCollectionsList, shpfilesList))

# Extract GLCM metrics
clipped_GLCM_collectionList = list(map(
  lambda collList: list(map(applyGLCM, collList)),
    clipped_mainBands_CollectionList))

# # Compute percentiles
percentiles = [5, 25, 50, 75, 100]
mainBands_percentile_collectionList = \
list(map(lambda collList: list(map(lambda collection: collection.reduce(
  ee.Reducer.percentile(percentiles)), collList)),
    clipped_mainBands_CollectionList))

glcmBands_percentile_collectionList = \
list(map(lambda collList: list(map(lambda collection: collection.reduce(
  ee.Reducer.percentile(percentiles)), collList)),
    clipped_GLCM_collectionList))

# Reduce each image in the imageCollections (with main bands) to mean
#  value over each field (for each year)
# This will produce a list of lists containing reduced featureCollections
reducedList_mainBands = list(map(
  lambda imgList, shp:percentile_imageReducer(
    imgList, ee.FeatureCollection(shp)),
       mainBands_percentile_collectionList, shpfilesList))    

# Reduce each image in the imageCollections (with GLCM bands)
#  to mean value over each field (for each year)
reducedList_glcmBands = list(map(
    lambda imgList, shp: percentile_imageReducer(
        imgList, ee.FeatureCollection(shp)),
    glcmBands_percentile_collectionList, shpfilesList))

# Extract band names to use in our dataframes
# The bands are identical for all years so we use the first year
#  imageCollection, [0]
# Main bands:
nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(),
                      mainBands_percentile_collectionList[0]))
mainBands = [name for sublist in nameLists for name in sublist]

# GLCM bands:
nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(),
                      glcmBands_percentile_collectionList[0]))
glcmBands = [name for sublist in nameLists for name in sublist]

# Convert each year's composites to a single dataframe 
# and put all the dataframes in a list a dataframe.

important_columns_names = ['pointID', 'CurrentCro', 'DateTime', 'PriorCropT', 
                           'ResidueCov', 'Tillage', 'WhereInRan']

metricBased_dataframeList_mainBands = eefeatureColl_to_Pandas_S1(
  reducedList_mainBands, mainBands, important_columns_names)

metricBased_dataframeList_glcm = eefeatureColl_to_Pandas_S1(
  reducedList_glcmBands, glcmBands, important_columns_names)

# Merge main and glcm bands for each year
allYears_metricBased_list = list(map(
    lambda mainband_df, glcmband_df: pd.concat(
        [mainband_df, glcmband_df], axis=1),
    metricBased_dataframeList_mainBands, metricBased_dataframeList_glcm))

# Remove duplicated columns
duplicated_cols_idx = [df.columns.duplicated()
                       for df in allYears_metricBased_list]
Sentinel_1_metricBased_list = list(map(
    lambda df, dup_idx: df.iloc[:, ~dup_idx], allYears_metricBased_list, duplicated_cols_idx))

print(Sentinel_1_metricBased_list[0].shape)
print(Sentinel_1_metricBased_list[1].shape)

# -

yearly_dfs_season = merge_cdl(Landsat_seasonBased_list, cdl_list)
yearly_dfs_metric = merge_cdl(Landsat_metricBased_list, cdl_list)


# +
import pandas as pd

path_to_save_batches = (
    "/Users/aminnorouzi/Library/CloudStorage/"
    "OneDrive-WashingtonStateUniversity(email.wsu.edu)/"
    "Ph.D/Projects/Tillage_Mapping/Data/field_level_data/"
    "mapping_data/"
)

cdl_2017 = pd.read_csv(path_to_save_batches + "/2017/cdl_all.csv")
seasonBased_2017_landsat = pd.read_csv(
    path_to_save_batches + "/2017/seasonBased_landsat.csv"
)
seasonBased_2017_sentinel1 = pd.read_csv(
    path_to_save_batches + "/2017/seasonBased_sentinel1.csv"
)
# -

cdl_2017
to_replace = {
    23: "Grain",
    31: "Canola",
    24: "Grain",
    51: "Legume",
    53: "Legume",
    61: "Fallow/Idle Cropland",
    52: "Legume",
    176: "Grassland/Pasture",
    35: "Mustard",
    21: "Grain",
    36: "Alfalfa",
}
cdl_2017["most_frequent_class"] = cdl_2017["most_frequent_class"].replace(to_replace)
cdl_filtered = cdl_2017[["pointID", "most_frequent_class"]]
cdl_df = cdl_filtered.loc[
    cdl_filtered["most_frequent_class"].isin(["Grain", "Canola", "Legume"])
]
cdl_df = cdl_df.rename(columns={"most_frequent_class": "ResidueType"})
cdl_df

seasonBased_2017_landsat_cdl = pd.merge(
    seasonBased_2017_landsat, cdl_df, on="pointID", how="left"
)

# +
seasonBased_2017_landsat_sentinel1 = pd.concat(
    [seasonBased_2017_landsat, seasonBased_2017_sentinel1.loc[:, "VH_S0":]], axis=1
)
df2017 = pd.merge(seasonBased_2017_landsat_sentinel1, cdl_df, on="pointID", how="left")

# Get the list of columns
cols = list(df2017.columns)

# Pop the last column
last_column = cols.pop(-1)  # -1 refers to the last item

# Insert the last column at the new position (index 2 for the third position, since Python is 0-indexed)
cols.insert(2, last_column)

# Reindex the DataFrame with the new column order
df2017 = df2017[cols]
df2017.to_csv(path_to_save_batches + "/2017/df_2017_test.csv")
