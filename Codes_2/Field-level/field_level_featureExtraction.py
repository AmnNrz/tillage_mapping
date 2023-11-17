# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
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
# # Mount google drive
# from google.colab import drive
# drive.mount('/content/drive')

# + [markdown] id="dl5KSrInfIGI"
# #### Functions

# + colab={"background_save": true} id="QaaLjXabmhWA"
#######################     Functions     ######################

# ///// Rename Landsat 8, 7 and 5 bands /////

def renameBandsL8(image):
    bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'];
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL'];
    return image.select(bands).rename(new_bands)

def renameBandsL7(image):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'];
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL'];
    return image.select(bands).rename(new_bands);

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
  qa = image.select('QA_PIXEL') ##substitiu a band FMASK
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
  new_bandS2 = ['B_S2', 'G_S2', 'R_S2', 'NIR_S2', 'SWIR1_S2', 'SWIR2_S2', 'evi_S2', 'gcvi_S2', 'ndvi_S2', 'sndvi_S2', 'ndti_S2', 'ndi5_S2', 'ndi7_S2', 'crc_S2', 'sti_S2', 'doy_S2'];
  new_bandS3 = ['B_S3', 'G_S3', 'R_S3', 'NIR_S3', 'SWIR1_S3', 'SWIR2_S3', 'evi_S3', 'gcvi_S3', 'ndvi_S3', 'sndvi_S3', 'ndti_S3', 'ndi5_S3', 'ndi7_S3', 'crc_S3', 'sti_S3', 'doy_S3'];

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
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1),
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS1))

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis()))

  collection_3 = orgCollection\
    .filterDate(
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1),
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 8, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS2))

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1).millis()))

  collection_4 = orgCollection\
    .filterDate(
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1),
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 12, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS3))

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
      df_j = geemap.ee_to_pandas(ee.FeatureCollection(f_j))  # Convert featureCollection to pandas dataframe
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
        Comp_S2 = ith_Collection.toList(ith_Collection.size()).get(2)
        Comp_S3 = ith_Collection.toList(ith_Collection.size()).get(3)

        bands_to_rename = ['VV_dB', 'VH_dB']
        new_bandS0 = ['VV_S0', 'VH_S0']
        new_bandS1 = ['VV_S1', 'VH_S1']
        new_bandS2 = ['VV_S2', 'VH_S2']
        new_bandS3 = ['VV_S3', 'VH_S3']

        composite_S0_renamed = ee.Image(Comp_S0).select(
            bands_to_rename).rename(new_bandS0)
        composite_S1_renamed = ee.Image(Comp_S1).select(
            bands_to_rename).rename(new_bandS1)
        composite_S2_renamed = ee.Image(Comp_S2).select(
            bands_to_rename).rename(new_bandS2)
        composite_S3_renamed = ee.Image(Comp_S3).select(
            bands_to_rename).rename(new_bandS3)

        renamedCollection = ee.ImageCollection.fromImages(
            [composite_S0_renamed, composite_S1_renamed, composite_S2_renamed, composite_S3_renamed])
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
      df_j = eefeaturecoll_to_pandas_manual(ee.FeatureCollection(f_j))
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
  new_bandS0 = ['VV_S0', 'VH_S0', 'doy_S0'];
  new_bandS1 = ['VV_S1', 'VH_S1', 'doy_S1'];
  new_bandS2 = ['VV_S2', 'VH_S2', 'doy_S2'];
  new_bandS3 = ['VV_S3', 'VH_S3', 'doy_S3'];

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
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1),
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS1))

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis()))

  collection_3 = orgCollection\
    .filterDate(
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1),
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 8, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS2))

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1).millis()))

  collection_4 = orgCollection\
    .filterDate(
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1),
      ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 12, 30)
    ).filterBounds(geometry)\
    .map(addDOY)\
    .map(lambda img: img.select(bands).rename(new_bandS3))

    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1).millis()))

  # Return a list of imageCollections

  return [collection_1, collection_2, collection_3, collection_4]

# -

# #### Imports

# +
######## imports #########
# consider a polygon that covers the study area (Whitman & Columbia counties)
geometry = ee.Geometry.Polygon(
    [[[-118.61039904725511, 47.40441980731236],
      [-118.61039904725511, 45.934467488469],
      [-116.80864123475511, 45.934467488469],
      [-116.80864123475511, 47.40441980731236]]], None, False)

geometry2 = ee.Geometry.Point([-117.10053796709163, 46.94957951590986]),

asset_folder = 'projects/ee-bio-ag-tillage/assets/tillmap_shp'
assets_list = ee.data.getList({'id': asset_folder})
shpfilesList = [i['id'] for i in assets_list]

path_to_data = ('/Users/aminnorouzi/Library/CloudStorage/'
                'OneDrive-WashingtonStateUniversity(email.wsu.edu)/Ph.D/'
                'Projects/Tillage_Mapping/Data/')

# path_to_data = ('/home/amnnrz/OneDrive - a.norouzikandelati/Ph.D/Projects/'
#                 'Tillage_Mapping/Data/')

startYear = 2021
endYear = 2023

# -

# # Download CDL data

# +
# Define date range
# ****** For 2021-2022 season, 2021 and for 2022-2023 season, 2022 cdl
# ****** data is downloaded and used as PriorCropType

# read shapefiles as geopandas dataframes and put them in a list
files = os.listdir(path_to_data +
                   'GIS_Data/final_shpfiles')

shapefile_names = [shp for shp in files if shp.endswith('.shp')]


geopandas_list = [gpd.read_file(
    path_to_data + 'GIS_Data/final_shpfiles/' + _
) for _ in shapefile_names]

# Load the USDA NASS CDL dataset
cdl = ee.ImageCollection("USDA/NASS/CDL") \
.filterDate(
    ee.Date.fromYMD(ee.Number(startYear), 1, 1),
    ee.Date.fromYMD(ee.Number(endYear), 12, 31)) \
.filterBounds(geometry)

def getMostFrequentClass(image, polygon):
    polygon = ee.Feature(polygon)
    # Clip the CDL image to the polygon
    cdlClipped = image.clip(polygon)

    # Calculate the histogram for the polygon
    histogram = cdlClipped.reduceRegion(**{
        'reducer': ee.Reducer.frequencyHistogram(),
        'geometry': polygon.geometry(),
        'scale': 30 # adjust based on CDL dataset's resolution
    }).get('cropland')

    # Convert the dictionary to key-value pairs
    keys = ee.Dictionary(histogram).keys()
    values = keys.map(
        lambda key:ee.Number(ee.Dictionary(histogram).get(key))
    )

    # Find the most frequent class
    max = values.reduce(ee.Reducer.max())
    indexMax = values.indexOf(max)
    mostFrequentClass = keys.get(indexMax)

    return polygon.set('most_frequent_class', mostFrequentClass)


# Initiate a list to store cdl data for each year
cdl_dataframes = {}

# filter colllection for the year
years = np.arange(startYear, endYear)

def cdl_dataframe_yith(year, shapefile):
    cdl_image = cdl.filterDate(
    ee.Date.fromYMD(ee.Number(int(year)), 1, 1),
    ee.Date.fromYMD(ee.Number(int(year)), 12, 31)) \
    .first()

    # Map the function across all polygons
    polygons = geemap.geopandas_to_ee(shapefile)
    polygonsList = eeList_to_pyList(polygons.toList(polygons.size()))

    cdl_feature_pylist = list(
        map(lambda pol: getMostFrequentClass(cdl_image, pol), polygonsList))
    
    cdl_feature_eelist = pyList_to_eeList(cdl_feature_pylist)

    
    cdl_y_data = ee.FeatureCollection(cdl_feature_eelist)
    return geemap.ee_to_pandas(cdl_y_data)

cdl_list = list(map(lambda year, shapefile: cdl_dataframe_yith(year, shapefile), years, geopandas_list))


# + [markdown] id="Xi8j9i9nSiW7"
# # Download Season-Based Landsat Data

# + colab={"background_save": true} id="1OJ1fUM1K-_S" outputId="6528614e-d813-455e-e2ff-42aa84416909"
#####################################################################
###################      Season-based Features      #################
#####################################################################
# Extract season-based features, using main bands, Indices and 
# their Gray-level Co-occurence Metrics (GLCMs)
# import USGS Landsat 8 Level 2, Collection 2, Tier 1

L8T1 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
L7T1 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")

L8_2122 = L8T1\
  .filter(ee.Filter.calendarRange(startYear, endYear, 'year'))\
  .map(lambda img: img.set('year', img.date().get('year')))\
  .map(lambda img: img.clip(geometry))

L7_2122 = L7T1\
  .filter(ee.Filter.calendarRange(startYear, endYear, 'year'))\
  .map(lambda img: img.set('year', img.date().get('year')))\
  .map(lambda img: img.clip(geometry))

# Apply scaling factor
L8_2122 = L8_2122.map(applyScaleFactors);
L7_2122 = L7_2122.map(applyScaleFactors);

# Rename bands
L8_2122 = L8_2122.map(renameBandsL8);
L7_2122 = L7_2122.map(renameBandsL7);

# Merge Landsat 7 and 8 collections
landSat_7_8 = ee.ImageCollection(L8_2122.merge(L7_2122))

# Apply NDVI mask
landSat_7_8 = landSat_7_8.map(addNDVI)

landSat_7_8 = landSat_7_8.map(lambda image: maskNDVI(image, threshold=0.3))

# Mask Clouds
landSat_7_8 = landSat_7_8.map(cloudMaskL8)

# Mask prercipitation > 3mm two days prior
# import Gridmet collection
GridMet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")\
                                  .filter(ee.Filter.date(
                                    ee.Date.fromYMD(ee.Number(startYear), 1, 1),
                                    ee.Date.fromYMD(ee.Number(endYear), 12, 30)))\
                                  .filterBounds(geometry)
landSat_7_8 = landSat_7_8.map(lambda image: MoistMask(image, GridMet))

# Add spectral indices to each in the collection as bands
landSat_7_8 = landSat_7_8.map(addIndices)

# Add Terrain variables (elevation, slope and aspect)
dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
slope = ee.Terrain.slope(dem)
aspect = ee.Terrain.aspect(dem)

# Merge Terrain variables with landsat image collection
landSat_7_8 = landSat_7_8.map(lambda image: image.addBands(dem)) \
              .map(lambda image: image.addBands(slope)) \
              .map(lambda image: image.addBands(aspect))

# Create season-based composites
# Specify time period
years = list(range(startYear, endYear));

# Create season-based composites for each year and put them in a list
yearlyCollectionsList = []
for y in years:
  yearlyCollectionsList = yearlyCollectionsList + [makeComposite(y, landSat_7_8)]
# Rename bands of each composite in each yearly collection
renamedCollectionList = renameComposites(yearlyCollectionsList)

  # Select shapefile
  # WSDA_featureCol = ee.FeatureCollection(
  #   'projects/ee-bio-ag-tillage/assets/' + f'{shp}')

# Clip each collection to the WSDA field boundaries
clipped_mainBands_CollectionList = list(map(
    lambda collection,shp: collection.map(
      lambda img: img.clip(ee.FeatureCollection(shp))),
        renamedCollectionList, shpfilesList))

# Extract GLCM metrics
clipped_GLCM_collectionList = list(map(applyGLCM, clipped_mainBands_CollectionList))


# Extract band names to use in our dataframes
# The bands are identical for all years so we use the first year imageCollection, [0]
# Main bands:
imageList = eeList_to_pyList(
  clipped_mainBands_CollectionList[0].toList(
    clipped_mainBands_CollectionList[0].size()))
nameLists = list(map(
  lambda img: ee.Image(img).bandNames().getInfo(), imageList))
mainBands = [name for sublist in nameLists for name in sublist]

# GLCM bands:
imageList = eeList_to_pyList(
  clipped_GLCM_collectionList[0].toList(clipped_GLCM_collectionList[0].size()))
nameLists = list(map(
  lambda img: ee.Image(img).bandNames().getInfo(), imageList))
glcmBands = [name for sublist in nameLists for name in sublist]

# Reduce each image in the imageCollections (with main bands) to
# mean value over each field (for each year). This will produce a list of 
# lists containing reduced featureCollections
reducedList_mainBands = list(map(
  lambda collection, shp: collectionReducer(collection, ee.FeatureCollection(
    shp)), clipped_mainBands_CollectionList, shpfilesList))   

# Reduce each image in the imageCollections (with GLCM bands) to mean value
# over each field (for each year)
reducedList_glcmBands = list(map(
  lambda collection, shp: collectionReducer(collection, ee.FeatureCollection(shp)),
    clipped_GLCM_collectionList, shpfilesList))

# Convert each year's composites to a single dataframe and put all 
# the dataframes in a list
important_columns_names = ['pointID', 'CurrentCro', 'DateTime',
                            'PriorCropT', 'ResidueCov', 'Tillage', 'WhereInRan']
seasonBased_dataframeList_mainBands = eefeatureColl_to_Pandas(
                    reducedList_mainBands, mainBands, important_columns_names)
seasonBased_dataframeList_glcm = eefeatureColl_to_Pandas(
                    reducedList_glcmBands, glcmBands, important_columns_names)

# Merge main and glcm bands for each year 
allYears_seasonBased_list = list(map(
    lambda mainband_df, glcmband_df: pd.concat( 
      [mainband_df, glcmband_df], axis=1),
        seasonBased_dataframeList_mainBands, seasonBased_dataframeList_glcm))

# Remove duplicated columns
duplicated_cols_idx = [df.columns.duplicated() for
                        df in allYears_seasonBased_list]
Landsat_seasonBased_list = list(map(
      lambda df, dup_idx: df.iloc[:, ~dup_idx],
        allYears_seasonBased_list, duplicated_cols_idx))

print(Landsat_seasonBased_list[0].shape)
print(Landsat_seasonBased_list[1].shape)


# Display on Map
# Map = geemap.Map()
# Map.setCenter(-117.100, 46.94, 7)
# Map.addLayer(ee.Image(clipped_mainBands_CollectionList[0].toList(
# clipped_mainBands_CollectionList[0].size(
# )).get(1)), {'bands': ['B4_S1', 'B3_S1', 'B2_S1'], max: 0.5, 'gamma': 2}, 'L8')
# Map
# + [markdown] id="DUhdHR8xIrUE"
# # Download Metric-Based Landsat Data

# + colab={"background_save": true} id="vrRY7E6NLhul"
from functools import reduce
###########################################################################
###################      Distribution-based Features      #################
###########################################################################
#### Extract distribution-based (metric-based) features using main bands, 
#### indices and Gray-level Co-occurence Metrics (GLCMs)

# Create metric composites
# Years
years = list(range(startYear, endYear));

# Create a list of lists of imageCollections. Each year would have n number 
# of imageCollection corresponding to the time periods specified
# for creating metric composites.
yearlyCollectionsList = []
for y in years:
  yearlyCollectionsList = yearlyCollectionsList + \
  [groupImages(y, landSat_7_8, geometry)]  # 'yearlyCollectionsList' is a Python list
# yearlyCollectionsList[0][0]

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

# -

# # Download Season-Based Sentinel-1 Data

# +

Sentinel_1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
    .filter(ee.Filter.calendarRange(startYear, endYear, 'year')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
    .map(lambda img: img.set('year', img.date().get('year')))\
    .map(lambda img: img.clip(geometry))

# Convert pixel values to logarithmic scale (decible scale)
def toDb(img):

    dB = ee.Image(10.0).multiply(img.log10()).toFloat()
    # Rename the bands for VV and VH
    bands = img.bandNames();
    newBands = bands.map(lambda band: ee.String(band).cat('_dB'))

    # Add dB bands and rename them
    imageWithDb = img.addBands(dB)
    renamedImage = imageWithDb.select(bands, newBands)

    return renamedImage


# Apply preprocessing and visualization
processedCollection = Sentinel_1 \
.map(toDb) \
.map(lambda img: img.select(['VV_dB', 'VH_dB']))

# # Display on map
# Map = geemap.Map(center=[46.94, -117.100], zoom=7)
# Map.addLayer(processedCollection, {
#              'bands': ['VV_dB', 'VH_dB'], 'min': -20, 'max': 0}, 'Sentinel-1')
# Map 

# Specify time period
years = list(range(startYear, endYear))

yearlyCollectionsList = []
for y in years:
  yearlyCollectionsList = yearlyCollectionsList + \
      [makeComposite(y, processedCollection)]
  
renamedCollectionList = renameComposites_S1(yearlyCollectionsList)

clipped_mainBands_CollectionList = list(map(
    lambda collection, shp: collection.map(
        lambda img: img.clip(ee.FeatureCollection(shp))),
    renamedCollectionList, shpfilesList))

clipped_mainBands_CollectionList

clipped_GLCM_collectionList = list(
    map(applyGLCM, clipped_mainBands_CollectionList))

clipped_GLCM_collectionList = list(
    map(applyGLCM, clipped_mainBands_CollectionList))

imageList = eeList_to_pyList(
    clipped_mainBands_CollectionList[0].toList(
        clipped_mainBands_CollectionList[0].size()))
nameLists = list(map(
    lambda img: ee.Image(img).bandNames().getInfo(), imageList))
mainBands = [name for sublist in nameLists for name in sublist]


# GLCM bands:
imageList = eeList_to_pyList(
    clipped_GLCM_collectionList[0].toList(clipped_GLCM_collectionList[0].size()))
nameLists = list(map(
    lambda img: ee.Image(img).bandNames().getInfo(), imageList))
glcmBands = [name for sublist in nameLists for name in sublist]

# Reduce each image in the imageCollections (with main bands) to
# mean value over each field (for each year). This will produce a list of
# lists containing reduced featureCollections
reducedList_mainBands = list(map(
    lambda collection, shp: collectionReducer(collection, ee.FeatureCollection(
        shp)), clipped_mainBands_CollectionList, shpfilesList))

# Reduce each image in the imageCollections (with GLCM bands) to mean value
# over each field (for each year)
reducedList_glcmBands = list(map(
    lambda collection, shp: collectionReducer(
        collection, ee.FeatureCollection(shp)),
    clipped_GLCM_collectionList, shpfilesList))

# Convert each year's composites to a single dataframe and put all
# the dataframes in a list
important_columns_names = ['pointID', 'CurrentCro', 'DateTime',
                           'PriorCropT', 'ResidueCov', 'Tillage', 'WhereInRan']
seasonBased_dataframeList_mainBands = eefeatureColl_to_Pandas_S1(
                    reducedList_mainBands, mainBands, important_columns_names)
seasonBased_dataframeList_glcm = eefeatureColl_to_Pandas_S1(
                    reducedList_glcmBands, glcmBands, important_columns_names)

# Merge main and glcm bands for each year
allYears_seasonBased_list = list(map(
    lambda mainband_df, glcmband_df: pd.concat(
        [mainband_df, glcmband_df], axis=1),
    seasonBased_dataframeList_mainBands, seasonBased_dataframeList_glcm))

# Remove duplicated columns
duplicated_cols_idx = [df.columns.duplicated() for
                       df in allYears_seasonBased_list]
Sentinel_1_seasonBased_list = list(map(
    lambda df, dup_idx: df.iloc[:, ~dup_idx],
    allYears_seasonBased_list, duplicated_cols_idx))

print(Sentinel_1_seasonBased_list[0].shape)
print(Sentinel_1_seasonBased_list[1].shape)

# -

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

