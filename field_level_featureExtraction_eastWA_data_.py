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
import geemap
import numpy as np
import geopandas as gpd
import pandas as pd
import os
import sys
# -

# These are hard coded here based on the number of fields in the shapefile.
num_batches = int(sys.argv[1])
batch_number = int(sys.argv[2])

from google.auth import default
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/service-account-file.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/a.norouzikandelati/Google_stuff/gee_credentials/clear-shadow-332006-e8d8faf764f0.json"
# Obtain credentials with the appropriate scope
# Obtain credentials with additional scope for Google Drive
credentials, _ = default(scopes=[
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive'
])
# # Initialize the Earth Engine API with the specified project
ee.Initialize(credentials=credentials, project='clear-shadow-332006')

# + [markdown] id="dl5KSrInfIGI"
# #### Functions

# + colab={"background_save": true} id="QaaLjXabmhWA"
#######################     Functions     ######################


# Make batches from a shapfile
def make_batch(shapfile, num_batches, batch_number):
  field_count = len(shapfile.pointID.unique())
  print (f"{field_count = }")

  batch_size = np.floor(field_count/num_batches)

  start_row = int((batch_number-1) * batch_size)
  end_row = int(start_row + batch_size - 1)

  if batch_number<num_batches:
      current_batch = shapfile.iloc[start_row:end_row]
  else:
      current_batch = shapfile.iloc[start_row:]

  shapefile_batch = current_batch.copy()
  shapefile_batch.reset_index(drop=True, inplace=True)
  print (shapefile_batch.shape)
  return shapefile_batch

# ///// Rename Landsat 8, 7 and 5 bands /////
def renameBandsL8(image):
    bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'];
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL'];
    return image.select(bands).rename(new_bands)

def renameBandsL7_and_5(image):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'];
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL'];
    return image.select(bands).rename(new_bands)

# ///// Apply scaling factor /////
def applyScaleFactors(image):
  opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);   # We are not using thermal bands.
  return image.addBands(opticalBands, None, True)\
              .addBands(thermalBands, None, True)

# ///// Computes spectral indices,  including EVI, GCVI, NDVI, SNDVI, NDTI, NDI5, NDI7, CRC, STI
# and adds them as bands to each image /////
def addIndices(image):
  # evi
  evi = image.expression('2.5 * (b("NIR") - b("R"))/(b("NIR") + 6 * b("R") - 7.5 * b("B") + 1)').rename('evi')

  # gcvi
  gcvi = image.expression('b("NIR")/b("G") - 1').rename('gcvi')

  # sndvi
  sndvi = image.expression('(b("NIR") - b("R"))/(b("NIR") + b("R") + 0.16)').rename('sndvi')

  # ndti
  ndti = image.expression('(b("SWIR1") - b("SWIR2"))/(b("SWIR1") + b("SWIR2"))').rename('ndti')

  # ndi5
  ndi5 = image.expression('(b("NIR") - b("SWIR1"))/(b("NIR") + b("SWIR1"))').rename('ndi5')

  # ndi7
  ndi7 = image.expression('(b("NIR") - b("SWIR2"))/(b("NIR") + b("SWIR2"))').rename('ndi7')

  # crc
  crc = image.expression('(b("SWIR1") - b("G"))/(b("SWIR1") + b("G"))').rename('crc')

  # sti
  sti = image.expression('b("SWIR1")/b("SWIR2")').rename('sti')

  return image.addBands(evi).addBands(gcvi)\
  .addBands(sndvi).addBands(ndti).\
  addBands(ndi5).addBands(ndi7).addBands(crc).addBands(sti)

# Mask cloud
def cloudMaskL8(image):
  qa = image.select('QA_PIXEL') 
  cloud1 = qa.bitwiseAnd(1<<3).eq(0)
  cloud2 = qa.bitwiseAnd(1<<9).eq(0)
  cloud3 = qa.bitwiseAnd(1<<4).eq(0)

  mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud1).updateMask(cloud2).updateMask(cloud3).updateMask(mask2).copyProperties(image, ["system:time_start"])

# ///// Add NDVI /////
def addNDVI(image):
    ndvi = image.normalizedDifference(['NIR', 'R']).rename('ndvi');
    return image.addBands(ndvi)

# ///// Mask NDVI /////
def maskNDVI (image, threshold):
  NDVI = image.select("ndvi")
  ndviMask = NDVI.lte(threshold);
  masked = image.updateMask(ndviMask)
  return masked

# ///// Mask pr>0.3 from GridMet image /////
def MoistMask(img, GridMet):
  # Find dates (2 days Prior) and filter Grid collection
  date_0 = img.date();
  date_next = date_0.advance(+1,"day");
  date_1 = date_0.advance(-1,"day");
  date_2 =date_0.advance(-2,"day");
  Gimg1 = GridMet.filterDate(date_2,date_1);
  Gimg2 = GridMet.filterDate(date_1,date_0);
  Gimg3 = GridMet.filterDate(date_0,date_next);

  # Sum of precipitation for all three dates
  GridMColl_123 = ee.ImageCollection(Gimg1.merge(Gimg2).merge(Gimg3));
  GridMetImgpr = GridMColl_123.select('pr');
  threeDayPrec = GridMetImgpr.reduce(ee.Reducer.sum());

  # Add threeDayPrec as a property to the image in the imageCollection
  img = img.addBands(threeDayPrec)
  # mask gridmet image for pr > 3mm
  MaskedGMImg = threeDayPrec.lte(3).select('pr_sum').eq(1);
  maskedLImg = img.updateMask(MaskedGMImg);
  return maskedLImg;

# ///// Make season-based composites /////
# Produces a list of imageCollections for each year. Each imageCollection contains the season-based composites for each year.
# Composites are created by taking the median of images in each group of the year.
def makeComposite (year, orgCollection):
    year = ee.Number(year)
    composite1 = orgCollection.filterDate(
        ee.Date.fromYMD(year, 9, 1),
        ee.Date.fromYMD(year, 12, 30)
      )\
      .median()\
      .set('system:time_start', ee.Date.fromYMD(year, 9, 1).millis())\
      .set('Date', ee.Date.fromYMD(year, 9, 1));

    composite2 = orgCollection\
      .filterDate(
        ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1),
        ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30)
      )\
      .median()\
      .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis())\
      .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1));

    composite3 = orgCollection\
      .filterDate(
        ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1),
        ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 8, 30)
      )\
      .median()\
      .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1).millis())\
      .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1));

    composite4 = orgCollection\
      .filterDate(
        ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1),
        ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 12, 30)
      )\
      .median()\
      .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1).millis())\
      .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1));

    # Return a collection of composites for the specific year
    return ee.ImageCollection(composite1)\
      .merge(ee.ImageCollection(composite2))\
      .merge(ee.ImageCollection(composite3))\
      .merge(ee.ImageCollection(composite4));

# ///// Add day of year (DOY) to each image as a band /////
def addDOY(img):
  doy = img.date().getRelative('day', 'year');
  doyBand = ee.Image.constant(doy).uint16().rename('doy')
  doyBand
  return img.addBands(doyBand)

# ///// Make metric-based imageCollections /////
# This groups images in a year and returns a list of imageCollections.


def groupImages(year, orgCollection, geometry):
  # This groups images and rename bands
  bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'evi', 'gcvi', 'ndvi', 'sndvi', 'ndti', 'ndi5', 'ndi7', 'crc', 'sti', 'doy'];
  new_bandS0 = ['B_S0', 'G_S0', 'R_S0', 'NIR_S0', 'SWIR1_S0', 'SWIR2_S0', 'evi_S0', 'gcvi_S0', 'ndvi_S0', 'sndvi_S0', 'ndti_S0', 'ndi5_S0', 'ndi7_S0', 'crc_S0', 'sti_S0', 'doy_S0'];
  new_bandS1 = ['B_S1', 'G_S1', 'R_S1', 'NIR_S1', 'SWIR1_S1', 'SWIR2_S1', 'evi_S1', 'gcvi_S1', 'ndvi_S1', 'sndvi_S1', 'ndti_S1', 'ndi5_S1', 'ndi7_S1', 'crc_S1', 'sti_S1', 'doy_S1'];


  year = ee.Number(year)
  collection_0 = orgCollection.filterDate(
      ee.Date.fromYMD(year, 9, 1),
      ee.Date.fromYMD(year, 12, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS0))

  collection_1 = orgCollection\
    .filterDate(
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 3, 1),
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS1))

  # Return a list of imageCollections
  return [collection_0, collection_1]


###### Add min NDTI
def add_minNDTI(year, orgCollection, geometry):
    # This groups images and rename bands
    bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'evi', 'gcvi', 'ndvi', 'sndvi', 'ndti', 'ndi5', 'ndi7', 'crc', 'sti', 'doy'];
    new_bandS0 = ['B_S0', 'G_S0', 'R_S0', 'NIR_S0', 'SWIR1_S0', 'SWIR2_S0', 'evi_S0', 'gcvi_S0', 'ndvi_S0', 'sndvi_S0', 'ndti_S0', 'ndi5_S0', 'ndi7_S0', 'crc_S0', 'sti_S0', 'doy_S0'];
    new_bandS1 = ['B_S1', 'G_S1', 'R_S1', 'NIR_S1', 'SWIR1_S1', 'SWIR2_S1', 'evi_S1', 'gcvi_S1', 'ndvi_S1', 'sndvi_S1', 'ndti_S1', 'ndi5_S1', 'ndi7_S1', 'crc_S1', 'sti_S1', 'doy_S1'];

    year = ee.Number(year)
    collection_0 = orgCollection.filterDate(
    ee.Date.fromYMD(year, 9, 1),
    ee.Date.fromYMD(year, 12, 30)
  ).filterBounds(geometry)\
  .map(addDOY)\
  .map(lambda img: img.select(bands).rename(new_bandS0))

    # Calculate min_NDTI for collection_1
    min_NDTI_S0 = (
      collection_0.select("ndti_S0").reduce(ee.Reducer.min()).rename("min_NDTI_S0")
  )
    # # Add min_NDTI to collection_0
    # collection_0 = collection_0.map(lambda img: img.addBands(min_NDTI_S0))
    # collection_0 = collection_0.map(lambda img: img.select("min_NDTI_S0"))

    collection_1 = (
      orgCollection.filterDate(
          ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 3, 1),
          ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30),
      )
      .filterBounds(geometry) \
      .map(addDOY)\
      .map(lambda img: img.select(bands).rename(new_bandS1))
  )

    # Calculate min_NDTI for collection_1
    min_NDTI_S1 = (
        collection_1.select("ndti_S1").reduce(ee.Reducer.min()).rename("min_NDTI_S1")
    )
    # # Add min_NDTI to collection_1
    # collection_1 = collection_1.map(lambda img: img.addBands(min_NDTI_S1))
    # collection_1 = collection_1.map(lambda img: img.select("min_NDTI_S1"))

    # Return a list of imageCollections
    return [min_NDTI_S0, min_NDTI_S1]


# ///// Rename the bands of each composite in the imageCollections associated with each year /////
def renameComposites(collectionList):
    renamedCollectionList = []
    for i in range(len(collectionList)):
        ith_Collection = collectionList[i]
        Comp_S0 = ith_Collection.toList(ith_Collection.size()).get(0)
        Comp_S1 = ith_Collection.toList(ith_Collection.size()).get(1)
        Comp_S2 = ith_Collection.toList(ith_Collection.size()).get(2)
        Comp_S3 = ith_Collection.toList(ith_Collection.size()).get(3)

        bandsNot_to_rename = ['elevation', 'slope', 'aspect']
        bands_to_rename = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'evi',
                           'gcvi', 'ndvi', 'sndvi', 'ndti', 'ndi5', 'ndi7', 'crc', 'sti']
        new_bandS0 = ['B_S0', 'G_S0', 'R_S0', 'NIR_S0', 'SWIR1_S0', 'SWIR2_S0',
                      'evi_S0', 'gcvi_S0', 'ndvi_S0', 'sndvi_S0', 'ndti_S0', 'ndi5_S0',
                      'ndi7_S0', 'crc_S0', 'sti_S0']
        new_bandS1 = ['B_S1', 'G_S1', 'R_S1', 'NIR_S1', 'SWIR1_S1', 'SWIR2_S1',
                      'evi_S1', 'gcvi_S1', 'ndvi_S1', 'sndvi_S1', 'ndti_S1', 'ndi5_S1',
                      'ndi7_S1', 'crc_S1', 'sti_S1']
        new_bandS2 = ['B_S2', 'G_S2', 'R_S2', 'NIR_S2', 'SWIR1_S2', 'SWIR2_S2',
                      'evi_S2', 'gcvi_S2', 'ndvi_S2', 'sndvi_S2', 'ndti_S2', 'ndi5_S2',
                      'ndi7_S2', 'crc_S2', 'sti_S2']
        new_bandS3 = ['B_S3', 'G_S3', 'R_S3', 'NIR_S3', 'SWIR1_S3', 'SWIR2_S3',
                      'evi_S3', 'gcvi_S3', 'ndvi_S3', 'sndvi_S3', 'ndti_S3', 'ndi5_S3',
                      'ndi7_S3', 'crc_S3', 'sti_S3']

        composite_S0_renamed = ee.Image(Comp_S0).select(
            bands_to_rename).rename(new_bandS0)
        composite_S1_renamed = ee.Image(Comp_S1).select(
            bands_to_rename).rename(new_bandS1)
        composite_S2_renamed = ee.Image(Comp_S2).select(
            bands_to_rename).rename(new_bandS2)
        composite_S3_renamed = ee.Image(Comp_S3).select(
            bands_to_rename).rename(new_bandS3)

        composite_S0_Notrenamed = ee.Image(Comp_S0).select(bandsNot_to_rename)

        composite_S0 = ee.Image.cat(
            [composite_S0_renamed, composite_S0_Notrenamed])

        renamedCollection = ee.ImageCollection.fromImages(
            [composite_S0, composite_S1_renamed, composite_S2_renamed, composite_S3_renamed])
        renamedCollectionList = renamedCollectionList + [renamedCollection]
    return renamedCollectionList

# ///// Convert GEE list (ee.list) to python list /////
def eeList_to_pyList(eeList):
  pyList = []
  for i in range(eeList.size().getInfo()):
    pyList = pyList + [eeList.get(i)]
  return pyList

# # ///// Convert python list to GEE list (ee.list)/////
# def pyList_to_eeList(pyList):
#   eeList = ee.List([])
#   for i in range(len(pyList)):
#     eeList = eeList.add(pyList[i])
#   return eeList

def pyList_to_eeList(pyList):
  return ee.List(pyList)

# ///// Function to reduce each image in a collection (with different band
# names for each image) to
# a median value (median value over each field geometry) /////
def collectionReducer(imgcollection, shp):
  imageList = eeList_to_pyList(imgcollection.toList(imgcollection.size()))
  return list(map(lambda img:ee.Image(img).reduceRegions(**{
                                                  'collection':shp,
                                                  'reducer':ee.Reducer.median(),
                                                  'scale': 1000

                                                }), imageList))

# ///// Function to reduce each percentile image (with different band names for each image) to
# a median value (median value over each field geometry) /////
def percentile_imageReducer(imageList, shp):
  return list(map(lambda img: ee.Image(img).reduceRegions(**{
      'reducer': ee.Reducer.median(),
      'collection': shp,
      'scale': 1000,
      'tileScale': 16
  }), imageList))

# ///// Function to create pandas dataframes from geographically (by field) reduced featureCollections  /////
# Arguments: 1) List of lists of featureCollections:
#                              [[y1_f0, y1_f1, y1_f2, y1_f3], [y2_f0, y2_f1, y2_f2, y2_f3], ..., [yn_f0, yn_f1, yn_f2, yn_f3]]
#                              y1_f0 : season 1 (or time period 1) of year 1 reduced composite
# Output: Lists of dataframes. Each dataframe is the derived data for each year.
def eefeatureColl_to_Pandas(yearlyList, bandNames, important_columns_names):
  dataList = []   # This list is going to contain dataframes for each year data
  for i in range(len(yearlyList)):
    year_i = pyList_to_eeList(yearlyList[i])
    important_columns = important_columns_names + bandNames

    df_yi = pd.DataFrame([])
    for j in range(year_i.length().getInfo()):
      f_j = year_i.get(j)  # Jth featureCollection (reduced composite data)
      df_j = geemap.ee_to_gdf(ee.FeatureCollection(f_j))  # Convert featureCollection to pandas dataframe
      df_j = df_j[df_j.columns[(df_j.columns).isin(important_columns)]]   # Pick needed columns
      df_yi = pd.concat([df_yi, df_j], axis=1)
    df_yi = df_yi.loc[:,~df_yi.columns.duplicated()]   # Drop repeated 'pointID' columns

    # reorder columns
    df_yi = df_yi[important_columns]
    
    # Move pointID column to first position
    pointIDColumn = df_yi.pop("pointID")
    df_yi.insert(0, "pointID", pointIDColumn)
    dataList = dataList + [df_yi]
  return dataList

# ///// Function to extract Gray-level Co-occurrence Matrix (GLCM) for each band in the composites  /////
# Input: an imageCollection containing the composites made for a year
# Output: List of imageCollections with GLCM bands.
def applyGLCM(coll):
  # Cast image values to a signed 32-bit integer.
  int32Coll = coll.map(lambda img: img.toInt32())
  glcmColl = int32Coll.map(lambda img: img.glcmTexture().set("system:time_start", img.date()))
  return glcmColl

## function to merge cdl wiht main data
def merge_cdl(maindf_list, cdl_list):
    '''
    Merge main dataframe and cdl
    
    '''
    cdl_list = [df.drop(columns='PriorCropT') for df in cdl_list]

    # Rename most_frequent_class to PriorCropT
    cdl_list = [df.rename(columns={'most_frequent_class': 'ResidueType'}) for
                        df in cdl_list]

    # Select just priorCropT and pointID
    cdl_list = [df[['pointID', 'ResidueType']].copy() for df in cdl_list]



    # Rename cdl labels or crop type
    replacement_dict = {'24': 'grain', '23': 'grain', '51': 'legume',
                        '51': 'legume', '31': 'canola', '53': 'legume',
                        '21': 'grain', '51': 'legume', '52': 'legume',
                        '28': 'grain'}

    # Just rename the three categories for grain, canola and legume
    cdl_list = [df.replace({'ResidueType':replacement_dict}) for df in cdl_list]
    cdl_list = [df.loc[df['ResidueType'].isin(['grain', 'legume', 'canola'])] for 
        df in cdl_list]

    # Make a list of tuples.Each tupple contain one seasonbased 
    # Corresponding to a year and one cdl dataframe
    sat_cdl = list(zip(maindf_list, cdl_list))

    # Merge cdl with main dataframe 
    final_dataframes_list = list(map(lambda df_tuple: pd.merge(
        df_tuple[0], df_tuple[1], on='pointID'), sat_cdl))

    # move ResidueType to the 4th column
    [df.insert(3, 'ResidueType', df.pop(df.columns[-1])) 
        for df in final_dataframes_list]

    return final_dataframes_list

def eefeaturecoll_to_pandas_manual(fc):
    features = fc.getInfo()['features']
    dict_list = []
    for f in features:
        attr = f['properties']
        dict_list.append(attr)
    df = pd.DataFrame(dict_list)
    return df

def renameComposites_S1(collectionList):
    renamedCollectionList = []
    for i in range(len(collectionList)):
        ith_Collection = collectionList[i]
        Comp_S0 = ith_Collection.toList(ith_Collection.size()).get(0)
        Comp_S1 = ith_Collection.toList(ith_Collection.size()).get(1)

        bands_to_rename = ['VV_dB', 'VH_dB']
        new_bandS0 = ['VV_S0', 'VH_S0']
        new_bandS1 = ['VV_S1', 'VH_S1']

        composite_S0_renamed = ee.Image(Comp_S0).select(
            bands_to_rename).rename(new_bandS0)
        composite_S1_renamed = ee.Image(Comp_S1).select(
            bands_to_rename).rename(new_bandS1)

        renamedCollection = ee.ImageCollection.fromImages(
            [composite_S0_renamed, composite_S1_renamed])
        renamedCollectionList = renamedCollectionList + [renamedCollection]
    return renamedCollectionList

def eefeatureColl_to_Pandas_S1(yearlyList, bandNames, important_columns_names):
  dataList = []   # This list is going to contain dataframes for each year data
  for i in range(len(yearlyList)):
    year_i = pyList_to_eeList(yearlyList[i])
    important_columns = important_columns_names + bandNames

    df_yi = pd.DataFrame([])
    for j in range(year_i.length().getInfo()):
      f_j = year_i.get(j)  # Jth featureCollection (reduced composite data)
      # Convert featureCollection to pandas dataframe
      df_j = geemap.ee_to_gdf(ee.FeatureCollection(f_j))
      df_j = df_j[df_j.columns[(df_j.columns).isin(
          important_columns)]]   # Pick needed columns
      df_yi = pd.concat([df_yi, df_j], axis=1)
    # Drop repeated 'pointID' columns
    df_yi = df_yi.loc[:, ~df_yi.columns.duplicated()]

    # reorder columns
    df_yi = df_yi[important_columns]

    # Move pointID column to first position
    pointIDColumn = df_yi.pop("pointID")
    df_yi.insert(0, "pointID", pointIDColumn)
    dataList = dataList + [df_yi]
  return dataList

def groupImages_S1(year, orgCollection, geometry):
    # This groups images and rename bands
    bands = ['VV_dB', 'VH_dB', 'doy'];
    new_bandS0 = ['VV_S0', 'VH_S0', 'doy_S0']
    new_bandS1 = ['VV_S1', 'VH_S1', 'doy_S1']

    year = ee.Number(year)
    collection_1 = orgCollection.filterDate(
      ee.Date.fromYMD(year, 9, 1),
      ee.Date.fromYMD(year, 12, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS0))

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(year, 9, 1).millis()))

    collection_2 = orgCollection\
    .filterDate(
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 3, 1),
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS1))

    # Return a list of imageCollections
    return [collection_1, collection_2]


# Function to add terrain variables (elevation, slope and aspect)
def add_terrain(image_list):
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    slope = ee.Terrain.slope(dem)
    aspect = ee.Terrain.aspect(dem)

    # Merge Terrain variables with landsat image collection
    return [ee.Image(image).addBands(dem).addBands(slope).addBands(aspect)
            for image in image_list
            ]

def clip_images(image_list, shp):
   return [ee.Image(image).clip(shp) for image in image_list]

def remove_doy(image_list):
  new_image_list = []
  for img in image_list:
    # Get the list of all band names
    all_band_names = img.bandNames()

    # Filter out the bands that start with 'doy'
    bands_to_keep = all_band_names.filter(ee.Filter.stringStartsWith('item', 'doy').Not())

    # Select only the bands you want to keep
    img_filtered = img.select(bands_to_keep)
    new_image_list.append(img_filtered)
  return new_image_list

# Function to add 'year' property to each image
def addYear(image):
    return image.set('year', ee.Date(image.get('system:time_start')).get('year'))



# -

# #### Imports

# +
######## imports #########
# consider a polygon that covers the study area (Whitman & Columbia counties)
geometry = ee.Geometry.Polygon(
        [[[-124.76076353098078, 49.26930796465981],
          [-124.76076353098078, 45.32287437432444],
          [-116.35622251535578, 45.32287437432444],
          [-116.35622251535578, 49.26930796465981]]], None, False)

path_to_data = ('/home/a.norouzikandelati/Projects/Tillage_mapping/Data/')

# Read shapefiles as geopandas dataframes and put them in a list
files = os.listdir(path_to_data + "shapefiles/2012_2017_2022/")
shapefile_names = [shp for shp in files if shp.endswith(".shp")]
# Sorting based on the year extracted from the filename
shapefile_names = sorted(shapefile_names, key=lambda x: int(x.split('_')[1].split('.')[0]))

geopandas_list = [
    gpd.read_file(path_to_data + "shapefiles/2012_2017_2022/" + _)
    for _ in shapefile_names
]


# Create batches
geopandas_list = [
        make_batch(
            shp, num_batches, batch_number)
              for shp in geopandas_list
]

shpfilesList = [
    geemap.geopandas_to_ee(shp)
     for shp in geopandas_list
     ]

years = [2011, 2016, 2021]
# Expand years to one year before and after each year
# This is done for groupImages() function which filter fall of year i and spring
# of year i+1 
expanded_years = sorted(list(set([year - 1 for year in years] + years + [year + 1 for year in years])))
# -

# # Download CDL data

# ### Delete all files from service account drive

# +
import io
import os
import time
import pandas as pd
import math
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

# Function to authenticate and create a drive service
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Path to your service account key file
SERVICE_ACCOUNT_FILE = (
    "/home/a.norouzikandelati/Google_stuff/gee_credentials/"
    "clear-shadow-332006-e8d8faf764f0.json"
)

# Define the scopes
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Authenticate and construct service
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
service = build("drive", "v3", credentials=credentials)


# Function to delete all files
def delete_all_files():
    results = (
        service.files()
        .list(pageSize=100, fields="nextPageToken, files(id, name)")
        .execute()
    )
    items = results.get("files", [])
    if not items:
        print("No files found.")
    else:
        for item in items:
            try:
                service.files().delete(fileId=item["id"]).execute()
                print(f"Deleted file: {item['name']} (ID: {item['id']})")
            except Exception as e:
                print(f"Failed to delete {item['name']} (ID: {item['id']}): {e}")


# Call the function to delete all files
delete_all_files()

# +
# Define date range
# ****** For 2021-2022 season, 2021 and for 2022-2023 season, 2022 CDL
# ****** data is downloaded and used as PriorCropType

# Load the USDA NASS CDL dataset
cdl = (
    ee.ImageCollection("USDA/NASS/CDL")
    .map(addYear)
    .filter(ee.Filter.inList('year', years))
    .filterBounds(geometry)
)


def authenticate_drive():
    # Path to your service account key file
    SERVICE_ACCOUNT_FILE = (
    "/home/a.norouzikandelati/Google_stuff/gee_credentials/"
    "clear-shadow-332006-e8d8faf764f0.json"
)
    # Define the scopes
    SCOPES = ['https://www.googleapis.com/auth/drive']

    # Authenticate and construct service
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)
    return service

# Function to download a file from Google Drive
def download_file(service, file_id, file_path):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    with open(file_path, 'wb') as f:
        f.write(fh.read())
    print('Download Complete')

# Function to find the file on Google Drive
def find_file(service, file_name):
    response = service.files().list(q=f"name='{file_name}'", spaces='drive', fields="nextPageToken, files(id, name)").execute()
    for file in response.get('files', []):
        return file.get('id')
    return None

# Function to delete a specific file by its ID
def delete_file(file_id):
    # Path to your service account key file
    SERVICE_ACCOUNT_FILE = (
    "/home/a.norouzikandelati/Google_stuff/gee_credentials/"
    "clear-shadow-332006-e8d8faf764f0.json"
)
    # Define the scopes
    SCOPES = ['https://www.googleapis.com/auth/drive']

    # Authenticate and construct service
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)

    # Delete the specific file
    try:
        service.files().delete(fileId=file_id).execute()
        print(f"Deleted file with ID: {file_id}")
    except Exception as e:
        print(f"Failed to delete file (ID: {file_id}): {e}")


# Function to initiate and monitor the export task
def export_data_to_drive(image, batch_collection, description, file_format,
                         file_prefix, folder):

    histogram = image.reduceRegions(
        collection=batch_collection,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=30
    )

    task = ee.batch.Export.table.toDrive(
        collection=histogram,
        description=description,
        fileFormat=file_format,
        fileNamePrefix=file_prefix,
        folder=folder,
    )
    task.start()

    while task.active():
        print('Polling for task (id: {}).'.format(task.id))
        time.sleep(30)

    if task.status()['state'] == 'COMPLETED':
        print('Export completed successfully!')
        service = authenticate_drive()
        file_name = f"{file_prefix}.csv"
        file_id = find_file(service, file_name)
        if file_id:
            download_file(service, file_id, file_name)
            df = pd.read_csv(file_name)
            delete_file(file_id)
        else:
            print('File not found.')
    else:
        print('Error with export:', task.status())

    return df


def processInBatches(image, polygonsList, batch_size, file_name, batch_number, year):
    num_batches = math.ceil(len(polygonsList) / batch_size)
    cdl_df = pd.DataFrame()
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        print((start, end))
        polygon_batch = polygonsList[start:end]
        polygon_batch = pyList_to_eeList(polygon_batch)
        batch_collection = ee.FeatureCollection(polygon_batch)

        # for j in range(polygon_batch.size().getInfo())[:3]:
        #     polygon = polygon_batch.get(j)
        batch_df = export_data_to_drive(
            image,
            batch_collection,
            # f"histogram_export_batch_{i+1}_polygon_{j+1}",
            f"{year}" + "_batch_number:" + f"{batch_number}"
            + "_" + f"cdl_histogram_export_batch_{i+1}",
            "CSV",
            # f"{file_name}" + f"histogram_export_batch_{i+1}_polygon_{j+1}",
            f"{file_name}" + f"histogram_export_batch_{i+1}",
            "CDL_data",
        )
        cdl_df = pd.concat([cdl_df, batch_df])
    return cdl_df


def cdl_dataframe_yith_batched(year, shapefile, batch_size, file_name, batch_number):
    cdl_image = cdl.filterDate(
        ee.Date.fromYMD(ee.Number(int(year)), 1, 1),
        ee.Date.fromYMD(ee.Number(int(year)), 12, 31),
    ).first()

    polygons = geemap.geopandas_to_ee(shapefile)
    polygonsList = eeList_to_pyList(polygons.toList(polygons.size()))

    return processInBatches(cdl_image, polygonsList, batch_size, file_name, batch_number, year)

def find_most_requesnt_crop(dict_str):
    dict = eval(dict_str.replace('=', ':'))
    max_key = max(dict, key=dict.get)
    return max_key

# Define your batch size
batch_size = 50  # Adjust this based on your needs
# Example of processing and exporting for each year and shapefile
cdl_list = []
for year, shapefile, shpfile_name in zip(years, geopandas_list, shapefile_names):
    shapefile_name =shpfile_name
    file_name = shapefile_name + f"{year}"
    downloaded_cdl = cdl_dataframe_yith_batched(year, shapefile, batch_size, file_name, batch_number)
    downloaded_cdl['year'] = year
    cdl_list.append(downloaded_cdl)

cdl_df = pd.DataFrame()
for df in cdl_list:
    df["most_frequent_crop"] = df["cropland"].apply(find_most_requesnt_crop)
    cdl_df = pd.concat([cdl_df, df[["year", "pointID", "most_frequent_crop"]]])

cdl_df.to_csv(
    path_to_data
    + "2012_2017_2022/cdl_data/"
    + "cdl_df_batch_"
    + f"{batch_number}"
    + ".csv")

# + [markdown] id="DUhdHR8xIrUE"
# # Download Metric-Based Landsat Data

# + colab={"background_save": true} id="vrRY7E6NLhul"
from functools import reduce
#####################################################################
###################      Season-based Features      #################
#####################################################################
# Extract season-based features, using main bands, Indices and
# their Gray-level Co-occurence Metrics (GLCMs)
# import USGS Landsat 8 Level 2, Collection 2, Tier 1

L8T1 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
L7T1 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
L5T1 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")

L8 = (
    L8T1
    .map(addYear)
    .filter(ee.Filter.inList('year', expanded_years))
    .map(lambda img: img.set("year", img.date().get("year")))
    .map(lambda img: img.clip(geometry))
)

L7 = (
    L7T1
    .map(addYear)
    .filter(ee.Filter.inList('year', expanded_years))
    .map(lambda img: img.set("year", img.date().get("year")))
    .map(lambda img: img.clip(geometry))
)

L5 = (
    L5T1
    .map(addYear)
    .filter(ee.Filter.inList('year', expanded_years))
    .map(lambda img: img.set("year", img.date().get("year")))
    .map(lambda img: img.clip(geometry))
)

# Apply scaling factor
L8 = L8.map(applyScaleFactors)
L7 = L7.map(applyScaleFactors)
L5 = L5.map(applyScaleFactors)

# Rename bands
L8 = L8.map(renameBandsL8)
L7 = L7.map(renameBandsL7_and_5)
L5 = L5.map(renameBandsL7_and_5)

# Merge Landsat 7 and 8 collections
landSat_5_7_8 = ee.ImageCollection(L8.merge(L7).merge(L5))


# Apply NDVI mask
landSat_5_7_8 = landSat_5_7_8.map(addNDVI)

landSat_5_7_8 = landSat_5_7_8.map(lambda image: maskNDVI(image, threshold=0.3))

# Mask Clouds
landSat_5_7_8 = landSat_5_7_8.map(cloudMaskL8)

# Mask prercipitation > 3mm two days prior
# Import GridMet collection and apply the filters
GridMet = (
    ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
    .map(addYear)
    .filter(ee.Filter.inList('year', expanded_years))  # Filter images by 'year' property
    .filterBounds(geometry)  # Keep your existing spatial filter
)

landSat_5_7_8 = landSat_5_7_8.map(lambda image: MoistMask(image, GridMet))

# Add spectral indices to each in the collection as bands
landSat_5_7_8 = landSat_5_7_8.map(addIndices)

###########################################################################
###################      Distribution-based Features      #################
###########################################################################
#### Extract distribution-based (metric-based) features using main bands,
#### indices and Gray-level Co-occurence Metrics (GLCMs)

# Create metric composites

# Create a list of lists of imageCollections. Each year would have n number
# of imageCollection corresponding to the time periods specified
# for creating metric composites.
yearlyCollectionsList = [groupImages(y, landSat_5_7_8, geometry) for y in years]

minNDTI_imageList = [add_minNDTI(y, landSat_5_7_8, geometry) for y in years]

minNDTI_terrain_yearcollection = [
    add_terrain(image_list) for image_list in minNDTI_imageList
]

# Clip each collection to the WSDA field boundaries
clipped_mainBands_CollectionList = list(map(
  lambda collList, shp: list(map(
    lambda collection: ee.ImageCollection(collection).map(
      lambda img: ee.Image(img).clip(shp)), collList)),
        yearlyCollectionsList, shpfilesList))

clipped_minNDTI_terrain_CollectionList = [
    clip_images(images, shp)
    for images, shp in zip(minNDTI_terrain_yearcollection, shpfilesList)
]

# Extract GLCM metrics
clipped_GLCM_collectionList = list(map(
  lambda collList: list(map(applyGLCM, collList)),
    clipped_mainBands_CollectionList))

# # Compute percentiles
percentiles = [0, 5, 25, 50, 75, 100]
mainBands_percentile_collectionList = \
list(map(lambda collList: list(map(lambda collection: collection.reduce(
  ee.Reducer.percentile(percentiles)), collList)),
    clipped_mainBands_CollectionList))

# Merge min_NDTI and terrain to main bands
merged_lists = []
for sublist1, sublist2 in zip(
    mainBands_percentile_collectionList, clipped_minNDTI_terrain_CollectionList
):
    merged_sublist = [ee.Image(img1).addBands(ee.Image(img2)) for img1, img2 in zip(sublist1, sublist2)]
    merged_lists.append(merged_sublist)

mainBands_percentile_collectionList = merged_lists

mainBands_percentile_collectionList = [
    remove_doy(image_list) for image_list in mainBands_percentile_collectionList
]

glcmBands_percentile_collectionList = \
list(map(lambda collList: list(map(lambda collection: collection.reduce(
  ee.Reducer.percentile(percentiles)), collList)),
    clipped_GLCM_collectionList))

glcmBands_percentile_collectionList = [
    remove_doy(image_list) for image_list in glcmBands_percentile_collectionList
]

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
nameLists = list(map(lambda img: img.bandNames().getInfo(),
                      mainBands_percentile_collectionList[0]))
mainBands = [name for sublist in nameLists for name in sublist]

# GLCM bands:
nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(),
                      glcmBands_percentile_collectionList[0]))
glcmBands = [name for sublist in nameLists for name in sublist]

# Convert each year's composites to a single dataframe
# and put all the dataframes in a list a dataframe.

important_columns_names = [
    "pointID",
    "ExactAcres",
    "County"
]

metricBased_dataframeList_mainBands = eefeatureColl_to_Pandas(
  reducedList_mainBands, mainBands, important_columns_names)

metricBased_dataframeList_glcm = eefeatureColl_to_Pandas(
  reducedList_glcmBands, glcmBands, important_columns_names)

# Merge main and glcm bands for each year
allYears_metricBased_list = list(map(
    lambda mainband_df, glcmband_df: pd.concat(
        [mainband_df, glcmband_df], axis=1),
    metricBased_dataframeList_mainBands, metricBased_dataframeList_glcm))

# Remove duplicated columns
duplicated_cols_idx = [df.columns.duplicated()
                       for df in allYears_metricBased_list]
Landsat_metricBased_list = list(map(
    lambda df, dup_idx: df.iloc[:, ~dup_idx], allYears_metricBased_list, duplicated_cols_idx))

print(Landsat_metricBased_list[0].shape)
print(Landsat_metricBased_list[1].shape)

Landsat_metricBased_list_with_year = []
for df, year in zip(Landsat_metricBased_list, years):
    df = df.copy()
    df.loc[:, "year"] = year + 1
    Landsat_metricBased_list_with_year.append(df)

Landsat_metricBased_df = pd.DataFrame()
for df in Landsat_metricBased_list_with_year:
    Landsat_metricBased_df = pd.concat([Landsat_metricBased_df, df])
year_column = Landsat_metricBased_df.pop('year')
Landsat_metricBased_df.insert(1, 'year', year_column)
# -

Landsat_metricBased_df.to_csv(
    path_to_data + "2012_2017_2022/landsat_data/"
    + "Landsat_metricBased_eastwa_"
    + f"{batch_number}"
    + ".csv"
)
