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
#     display_name: tillmap
#     language: python
#     name: python3
# ---

# +
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

# +
######## imports #########
# consider a polygon that covers the study area (Whitman & Columbia counties)
geometry = ee.Geometry.Polygon(
    [[[-118.61039904725511, 47.40441980731236],
      [-118.61039904725511, 45.934467488469],
      [-116.80864123475511, 45.934467488469],
      [-116.80864123475511, 47.40441980731236]]], None, False)

geometry3 = ee.Geometry.Polygon(
    [[[-125.43652987007123, 48.930252835297736],
      [-125.43652987007123, 41.55030980517791],
      [-110.27539705757123, 41.55030980517791],
      [-110.27539705757123, 48.930252835297736]]], None, False)

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


# +
def makeComposite2(year, orgCollection):
    year = ee.Number(year)
    composite1 = orgCollection.filterDate(
        ee.Date.fromYMD(year, 9, 1),
        ee.Date.fromYMD(year, 12, 30)
    )\
        .median()\
        .set('system:time_start', ee.Date.fromYMD(year, 9, 1).millis())\
        .set('Date', ee.Date.fromYMD(year, 9, 1))

    composite2 = orgCollection\
        .filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30)
        )\
        .median()\
        .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis())\
        .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1))

    composite3 = orgCollection\
        .filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 8, 30)
        )\
        .median()\
        .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1).millis())\
        .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1))

    composite4 = orgCollection\
        .filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 12, 30)
        )\
        .median()\
        .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1).millis())\
        .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1))

    # Return a collection of composites for the specific year
    return ee.ImageCollection(composite1)\
        .merge(ee.ImageCollection(composite2))\
        .merge(ee.ImageCollection(composite3))\
        .merge(ee.ImageCollection(composite4))




#################################################################
#################################################################
#################################################################


def makeComposite(year, orgCollection):
    year = ee.Number(year)
    composite1 = orgCollection.filterDate(
        ee.Date.fromYMD(year, 9, 1),
        ee.Date.fromYMD(year, 12, 30)
    )\
        .median()\
        .set('system:time_start', ee.Date.fromYMD(year, 9, 1).millis())\
        .set('Date', ee.Date.fromYMD(year, 9, 1))

    composite2 = orgCollection\
        .filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30)
        )\
        .median()\
        .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis())\
        .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1))

    composite3 = orgCollection\
        .filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 8, 30)
        )\
        .median()\
        .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1).millis())\
        .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1))

    composite4 = orgCollection\
        .filterDate(
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1),
            ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 12, 30)
        )\
        .median()\
        .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1).millis())\
        .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1))

    # Return a collection of composites for the specific year
    return ee.ImageCollection(composite1)\
        .merge(ee.ImageCollection(composite2))\
        .merge(ee.ImageCollection(composite3))\
        .merge(ee.ImageCollection(composite4))


def renameComposites(collectionList):
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


def applyGLCM(coll):
  # Cast image values to a signed 32-bit integer.
  int32Coll = coll.map(lambda img: img.toInt32())
  glcmColl = int32Coll.map(
      lambda img: img.glcmTexture().set("system:time_start", img.date()))
  return glcmColl

# ///// Convert GEE list (ee.list) to python list /////


def eeList_to_pyList(eeList):
  pyList = []
  for i in range(eeList.size().getInfo()):
    pyList = pyList + [eeList.get(i)]
  return pyList


def collectionReducer(imgcollection, shp):
  imageList = eeList_to_pyList(imgcollection.toList(imgcollection.size()))
  return list(map(lambda img: ee.Image(img).reduceRegions(**{
      'collection': shp,
      'reducer': ee.Reducer.median(),
      'scale': 1000

  }), imageList))

def ee_featurecoll_to_pandas(fc):
    features = fc.getInfo()['features']
    dict_list = []
    for f in features:
        attr = f['properties']
        dict_list.append(attr)
    df = pd.DataFrame(dict_list)
    return df

def eefeatureColl_to_Pandas(yearlyList, bandNames, important_columns_names):
  dataList = []   # This list is going to contain dataframes for each year data
  for i in range(len(yearlyList)):
    year_i = pyList_to_eeList(yearlyList[i])
    important_columns = important_columns_names + bandNames

    df_yi = pd.DataFrame([])
    for j in range(year_i.length().getInfo()):
      f_j = year_i.get(j)  # Jth featureCollection (reduced composite data)
      # Convert featureCollection to pandas dataframe
      df_j = ee_featurecoll_to_pandas(ee.FeatureCollection(f_j))
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


def pyList_to_eeList(pyList):
  eeList = ee.List([])
  for i in range(len(pyList)):
    eeList = eeList.add(pyList[i])
  return eeList


def addDOY(img):
  doy = img.date().getRelative('day', 'year');
  doyBand = ee.Image.constant(doy).uint16().rename('doy')
  doyBand
  return img.addBands(doyBand)

def groupImages(year, orgCollection, geometry):
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


def percentile_imageReducer(imageList, shp):
  return list(map(lambda img: ee.Image(img).reduceRegions(**{
      'reducer': ee.Reducer.median(),
      'collection': shp,
      'scale': 1000,
      'tileScale': 16
  }), imageList))


# +
import geemap
    
Sentinel_1 = ee.ImageCollection("COPERNICUS/S1_GRD") \
    .filter(ee.Filter.calendarRange(startYear, endYear, 'year')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
    .map(lambda img: img.set('year', img.date().get('year')))\
    .map(lambda img: img.clip(geometry3))

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
  
renamedCollectionList = renameComposites(yearlyCollectionsList)

renamedCollectionList[0]


# +
clipped_mainBands_CollectionList = list(map(
    lambda collection, shp: collection.map(
        lambda img: img.clip(ee.FeatureCollection(shp))),
    renamedCollectionList, shpfilesList))

clipped_mainBands_CollectionList

clipped_GLCM_collectionList = list(
    map(applyGLCM, clipped_mainBands_CollectionList))


# +
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
seasonBased_list = list(map(
    lambda df, dup_idx: df.iloc[:, ~dup_idx],
    allYears_seasonBased_list, duplicated_cols_idx))

print(seasonBased_list[0].shape)
print(seasonBased_list[1].shape)

# -

seasonBased_list[1].columns

# +
from functools import reduce
###########################################################################
###################      Distribution-based Features      #################
###########################################################################

# Create metric composites
# Years
years = list(range(startYear, endYear));

# Create a list of lists of imageCollections. Each year would have n number 
# of imageCollection corresponding to the time periods specified
# for creating metric composites.
yearlyCollectionsList = []
for y in years:
  yearlyCollectionsList = yearlyCollectionsList + \
  [groupImages(y, processedCollection, geometry)]  # 'yearlyCollectionsList' is a Python list
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
metricBased_list = list(map(
    lambda df, dup_idx: df.iloc[:, ~dup_idx], allYears_metricBased_list, duplicated_cols_idx))

print(metricBased_list[0].shape)
print(metricBased_list[1].shape)

# -


