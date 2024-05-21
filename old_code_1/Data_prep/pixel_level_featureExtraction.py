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

# + id="whtD--m_FObX" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1681176983583, "user_tz": 420, "elapsed": 51962, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}} outputId="199c206e-274f-45e0-843a-ad76a2a44796"
# # !pip install --upgrade requests
# import os
# os.environ["EE_FORCE_JSON"] = "true"
# # Your code goes here

# # Initialize GEE python API
# # !earthengine authenticate
import ee

# Authenticate gee
ee.Authenticate()

# Initialize the library.
ee.Initialize()

# !pip install geemap
# !pip install geopandas 

import ee
import geemap
import numpy as np 
import geopandas as gpd
import pandas as pd

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 18157, "status": "ok", "timestamp": 1681177001735, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="nFzq5DXISHSg" outputId="7eb5fc63-3827-4373-e210-4de8f7654415"
from google.colab import drive
drive.mount('/content/drive')

# + [markdown] id="zNHfo5P4FUfQ"
# # Download raw and derived indices data for residue and crop-type classification from GEE imagery data:

# + [markdown] id="MWRiGjZRHCOf"
# #### Imports

# + id="RkSJ3M7GG_pD" executionInfo={"status": "ok", "timestamp": 1681177001736, "user_tz": 420, "elapsed": 7, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}}
######## imports #########
# Import WSDA polygons of surveyed fields 
# consider a polygon that covers the study area (Whitman & Columbia counties)
geometry = ee.Geometry.Polygon(
        [[[-118.61039904725511, 47.40441980731236],
          [-118.61039904725511, 45.934467488469],
          [-116.80864123475511, 45.934467488469],
          [-116.80864123475511, 47.40441980731236]]], None, False)
geometry2 = ee.Geometry.Point([-117.10053796709163, 46.94957951590986]),
WSDA_featureCol = ee.FeatureCollection("projects/ee-bio-ag-tillage/assets/2021_2022_pols")

#import USGS Landsat 8 and 7 Level 2, Collection 2, Tier 1
L8T1 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
L7T1 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")

# + id="K76xDXzHhbWb" executionInfo={"status": "ok", "timestamp": 1681177010149, "user_tz": 420, "elapsed": 8418, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}}
# shapeFile_path = r"G:\My Drive\PhD\Bio_AgTillage\01. Codes\pipeline_tillageClassifier\vsCode Integration\Data\WSDA_checkedForPins.dbf"
shapeFile_path = "/content/drive/MyDrive/P.h.D_Projects/Tillage_Mapping/Data/GIS_Data/WSDA_survey/WSDA_checkedForPins.dbf"
WSDA_featureCol_goepd = gpd.read_file(shapeFile_path, crs='EPSG:4326')
WSDA_featureCol = geemap.geopandas_to_ee(WSDA_featureCol_goepd)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2222, "status": "ok", "timestamp": 1681177012368, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="-wfrgVd2I9Lt" outputId="64548101-4ee3-42ea-92c4-7e24ca995111"
features_geopandasList = [WSDA_featureCol_goepd.loc[i:i] for i in range(WSDA_featureCol_goepd.shape[0])]
features_geopandasList[0]["pointID"]

# + id="IUyaOcbwKw56"
emptyData_feature_pointIDList = [int(features_geopandasList[i]["pointID"].values) for i in n]
emptyData_feature_pointIDList
# int(emptyData_featureList[0].index.values)

# + [markdown] id="dl5KSrInfIGI"
# #### Functions

# + executionInfo={"elapsed": 164, "status": "ok", "timestamp": 1681177126184, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="QaaLjXabmhWA"
#######################     Functions     ######################

# ///// Functions to rename Landsat 8, 7 and 5 bands /////

def renameBandsL8(image):
    bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL'];
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL'];
    return image.select(bands).rename(new_bands);
  
def renameBandsL7(image):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'];
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL'];
    return image.select(bands).rename(new_bands);
  
def renameBandsL5(image):
    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL'];
    new_bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'QA_PIXEL'];
    return image.select(bands).rename(new_bands);

# ///// Function to apply scaling factor /////
def applyScaleFactors(image):
  opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);   # We are not using thermal bands. 
  return image.addBands(opticalBands, None, True)\
              .addBands(thermalBands, None, True)

# ///// Function that computes spectral indices,  including EVI, GCVI, NDVI, SNDVI, NDTI, NDI5, NDI7, CRC, STI 
# and adds them as bands to each image /////
def addIndices(image):
  # evi
  evi = image.expression('2.5 * (b("NIR") - b("R"))/(b("NIR") + 6 * b("R") - 7.5 * b("B") + 1)').rename('evi')
  
  # gcvi
  gcvi = image.expression('b("NIR")/b("G") - 1').rename('gcvi')
  
  # ndvi
  ndvi = image.normalizedDifference(['NIR', 'R']).rename('ndvi');
  
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
  .addBands(ndvi).addBands(sndvi).addBands(ndti).\
  addBands(ndi5).addBands(ndi7).addBands(crc).addBands(sti)

def cloudMaskL8(image):
  qa = image.select('QA_PIXEL') ##substitiu a band FMASK
  cloud1 = qa.bitwiseAnd(1<<3).eq(0)
  cloud2 = qa.bitwiseAnd(1<<9).eq(0)
  cloud3 = qa.bitwiseAnd(1<<4).eq(0)

  mask2 = image.mask().reduce(ee.Reducer.min());
  return image.updateMask(cloud1).updateMask(cloud2).updateMask(cloud3).updateMask(mask2).copyProperties(image, ["system:time_start"])

# ///// Function to mask NDVI /////
def maskNDVI (image):
  NDVI_threshold = 0.3
  NDVI = image.select("ndvi")
  ndviMask = NDVI.lte(NDVI_threshold);
  masked = image.updateMask(ndviMask)
  return masked

# ///// Function to add NDVI /////
def addNDVI(image):
    ndvi = image.normalizedDifference(['NIR', 'R']).rename('ndvi');
    return image.addBands(ndvi)

# ///// Function to mask pr>0.3 from GridMet image /////
# import Gridmet collection
GridMet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")\
                                  .filter(ee.Filter.date('2021-1-1','2022-12-30'))\
                                  .filterBounds(geometry);
def MoistMask(img):
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

# ///// Function to make season-based composites ///// 
# It will produce a list of imageCollections for each year. Each imageCollection contains the season-based composites for each year.
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
        ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 3, 1), 
        ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30)
      )\
      .median()\
      .set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis())\
      .set('Date', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1));

    # Return a collection of composites for the specific year   
    return ee.ImageCollection(composite1)\
      .merge(ee.ImageCollection(composite2))

# ///// Function to add day of year (DOY) to each image as a band /////
def addDOY(img):
  doy = img.date().getRelative('day', 'year');
  doyBand = ee.Image.constant(doy).uint16().rename('doy')
  doyBand
  return img.addBands(doyBand)

# ///// Function to make metric-based imageCollections ///// 
# This groups images in a year and returns a list of imageCollections.
def groupImages(year, orgCollection):
# This groups images and rename bands
  bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'evi', 'gcvi', 'ndvi', 'sndvi', 'ndti', 'ndi5', 'ndi7', 'crc', 'sti', 'doy'];
  new_bandS0 = ['B_fall', 'G_fall', 'R_fall', 'NIR_fall', 'SWIR1_fall', 'SWIR2_fall', 'evi_fall', 'gcvi_fall', 'ndvi_fall', 'sndvi_fall', 'ndti_fall', 'ndi5_fall', 'ndi7_fall', 'crc_fall', 'sti_fall', 'doy_fall'];
  new_bandS1 = ['B_spring', 'G_spring', 'R_spring', 'NIR_spring', 'SWIR1_spring', 'SWIR2_spring', 'evi_spring', 'gcvi_spring', 'ndvi_spring', 'sndvi_spring', 'ndti_spring', 'ndi5_spring', 'ndi7_spring', 'crc_spring', 'sti_spring', 'doy_spring'];
  # new_bandS2 = ['B_S2', 'G_S2', 'R_S2', 'NIR_S2', 'SWIR1_S2', 'SWIR2_S2', 'evi_S2', 'gcvi_S2', 'ndvi_S2', 'sndvi_S2', 'ndti_S2', 'ndi5_S2', 'ndi7_S2', 'crc_S2', 'sti_S2', 'doy_S2'];
  # new_bandS3 = ['B_S3', 'G_S3', 'R_S3', 'NIR_S3', 'SWIR1_S3', 'SWIR2_S3', 'evi_S3', 'gcvi_S3', 'ndvi_S3', 'sndvi_S3', 'ndti_S3', 'ndi5_S3', 'ndi7_S3', 'crc_S3', 'sti_S3', 'doy_S3'];
  
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
    
    # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis()))

  # Return a list of imageCollections
  return [collection_1, collection_2]

# ///// Function to rename the bands of each composite in the imageCollections associated with each year ///// 
def renameComposites(collectionList):
  renamedCollectionList = []
  for i in range(len(collectionList)):
    ith_Collection = collectionList[i]
    Comp_S0 = ith_Collection.toList(ith_Collection.size()).get(0);
    Comp_S1 = ith_Collection.toList(ith_Collection.size()).get(1);
    # Comp_S2 = ith_Collection.toList(ith_Collection.size()).get(2);
    # Comp_S3 = ith_Collection.toList(ith_Collection.size()).get(3);

    bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'evi', 'gcvi', 'ndvi', 'sndvi', 'ndti', 'ndi5', 'ndi7', 'crc', 'sti'];
    new_bandS0 = ['B_fall', 'G_fall', 'R_fall', 'NIR_fall', 'SWIR1_fall', 'SWIR2_fall', 'evi_fall', 'gcvi_fall', 'ndvi_fall', 'sndvi_fall', 'ndti_fall', 'ndi5_fall', 'ndi7_fall', 'crc_fall', 'sti_fall'];
    new_bandS1 = ['B_spring', 'G_spring', 'R_spring', 'NIR_spring', 'SWIR1_spring', 'SWIR2_spring', 'evi_spring', 'gcvi_spring', 'ndvi_spring', 'sndvi_spring', 'ndti_spring', 'ndi5_spring', 'ndi7_spring', 'crc_spring', 'sti_spring'];
   # new_bandS2 = ['B_S2', 'G_S2', 'R_S2', 'NIR_S2', 'SWIR1_S2', 'SWIR2_S2', 'evi_S2', 'gcvi_S2', 'ndvi_S2', 'sndvi_S2', 'ndti_S2', 'ndi5_S2', 'ndi7_S2', 'crc_S2', 'sti_S2'];
    # new_bandS3 = ['B_S3', 'G_S3', 'R_S3', 'NIR_S3', 'SWIR1_S3', 'SWIR2_S3', 'evi_S3', 'gcvi_S3', 'ndvi_S3', 'sndvi_S3', 'ndti_S3', 'ndi5_S3', 'ndi7_S3', 'crc_S3', 'sti_S3'];

    composite_S0 = ee.Image(Comp_S0).select(bands).rename(new_bandS0)
    composite_S1 = ee.Image(Comp_S1).select(bands).rename(new_bandS1)
    # composite_S2 = ee.Image(Comp_S2).select(bands).rename(new_bandS2)
    # composite_S3 = ee.Image(Comp_S3).select(bands).rename(new_bandS3)

    renamedCollection = ee.ImageCollection.fromImages([composite_S0, composite_S1]);
    renamedCollectionList = renamedCollectionList + [renamedCollection]
    return renamedCollectionList

# ///// Function to convert GEE list (ee.list) to python list ///// 
def eeList_to_pyList(eeList):
  pyList = []
  for i in range(eeList.size().getInfo()):
    pyList = pyList + [eeList.get(i)]
  return pyList

# ///// Function to convert python list to GEE list (ee.list)///// 
def pyList_to_eeList(pyList):
  eeList = ee.List([])
  for i in range(len(pyList)):
    eeList = eeList.add(pyList[i])
  return eeList

# ///// Functions to extract pixel values of each image (season-based composite) and save them to drive.
# ///// This function loops over each each image and each polygon.
# For main bands:
def mainBand_pixelExtractor(imgcollection):
  imageList = eeList_to_pyList(imgcollection.toList(imgcollection.size()))
  # Get year of the imageCollection to use for saving names
  year = imgcollection.first().date().format("YYYY-MM-dd").getInfo().split("-")[0]
  for idx, img in enumerate(imageList): 
    # idx = idx + 1
    imageList_names = []
    first_BandName = ee.Image(imageList[idx]).bandNames().get(0).getInfo() # Used for saving each composite with it's related name "B_Sx" where x = 0, 1 or more"
    imageList_names = imageList_names + [first_BandName]
    print(first_BandName)
    for idx, f in enumerate(features_geopandasList):
      # idx = idx + 726
      pixel_featureCollection = ee.Image(img).sampleRegions(**{
                                                  'collection': geemap.geopandas_to_ee(features_geopandasList[idx]),
                                                  'scale': 10,
                                                  'geometries': True,
                                                  'tileScale': 16
                                                                    })
      task = ee.batch.Export.table.toDrive(**{
                                        'collection': pixel_featureCollection,
                                        'description': year + "_" + str(int(year) + 1) + '_mainBand_' + first_BandName.split("_")[1] + '_polygon' + f'{features_geopandasList[idx]}',
                                        'folder': 'seasonBased_Pixel_level_mainBands_TillageData_emptyData',
                                        'fileNamePrefix': year + "_" + str(int(year) + 1) + '_mainBand_' + first_BandName.split("_")[1] + '_polygon' + f'{features_geopandasList[idx]}',
                                        'fileFormat': 'CSV'})
      task.start()
      # import time 
      # while task.active():
      #   print('Polling for task (id: {}). Still breathing'.format(task.id))
      #   time.sleep(30)
  return imageList_names


# For glcm bands:
def glcmBand_pixelExtractor(imgcollection):
  imageList = eeList_to_pyList(imgcollection.toList(imgcollection.size()))
  # Get year of the imageCollection to use for saving names
  year = imgcollection.first().date().format("YYYY-MM-dd").getInfo().split("-")[0]
  for idx, img in enumerate(imageList[1:]): 
    idx = idx + 1
    imageList_names = []
    first_BandName = ee.Image(imageList[idx]).bandNames().get(0).getInfo() # Used for saving each composite with it's related name "B_Sx" where x = 0, 1 or more"
    imageList_names = imageList_names + [first_BandName]
    print(first_BandName)
    for idx, f in enumerate(features_geopandasList):
      # idx = idx + 535
      pixel_featureCollection = ee.Image(img).sampleRegions(**{
                                                  'collection': geemap.geopandas_to_ee(features_geopandasList[idx]),
                                                  'scale': 10,
                                                  'geometries': True,
                                                  'tileScale': 16
                                                                    })
      task = ee.batch.Export.table.toDrive(**{
                                        'collection': pixel_featureCollection,
                                        'description': year + "_" + str(int(year) + 1) + '_glcmBand_' + first_BandName.split("_")[1] + '_polygon' + f'{idx}',
                                        'folder': 'seasonBased_pixel_level_glcmBands_TillageData',
                                        'fileNamePrefix': year + "_" + str(int(year) + 1) + '_glcmBand_' + first_BandName.split("_")[1] + '_polygon' + f'{idx}',
                                        'fileFormat': 'CSV'})
      task.start()
      # import time 
      # while task.active():
      #   print('Polling for task (id: {}). Still breathing'.format(task.id))
      #   time.sleep(30)
  return imageList_names




# ///// Functions to extract pixel values of each image (distribution-based composite) and save them to drive.
# ///// This function loops over each each image and each polygon.
# For main bands:
def metricBased_mainBand_pixelExtractor(imgList, percentile):
  # Get year of the imageCollection to use for saving names
  year = imgList[0].date().format("YYYY-MM-dd").getInfo().split("-")[0]
  for idx, img in enumerate(imgList[:1]):
    # idx = idx + 1 
    imageList_names = []
    first_BandName = ee.Image(imgList[idx]).bandNames().get(0).getInfo() # Used for saving each composite with it's related name, e.g. "B_fall_p5"
    imageList_names = imageList_names + [first_BandName]
    print(first_BandName)
    for idx, f in enumerate(features_geopandasList[274:374]):
      idx = idx + 274
      pixel_featureCollection = ee.Image(img).sampleRegions(**{
                                                  'collection': geemap.geopandas_to_ee(features_geopandasList[idx]),
                                                  'scale': 10,
                                                  'geometries': True,
                                                  'tileScale': 16
                                                                    })
      task = ee.batch.Export.table.toDrive(**{
                                        'collection': pixel_featureCollection,
                                        'description': year + "_" + str(int(year) + 1) + '_mainBand_' + first_BandName.split("_")[1] + "_P" + str(percentile) + '_polygon' + f'{idx}',
                                        'folder': 'Distribution-Based_mainBands_TillageData',
                                        'fileNamePrefix': year + "_" + str(int(year) + 1) + '_mainBand_' + first_BandName.split("_")[1] + "_P" +str(percentile) + '_polygon' + f'{idx}',
                                        'fileFormat': 'CSV'})
      task.start()
      # import time 
      # while task.active():
      #   print('Polling for task (id: {}). Still breathing'.format(task.id))
      #   time.sleep(30)
  return imageList_names


# For glcm bands:
def metricBased_glcmBand_pixelExtractor(imgList, percentile):
  # Get year of the imageCollection to use for saving names
  year = imgList[0].date().format("YYYY-MM-dd").getInfo().split("-")[0]
  for img_idx, img in enumerate(imgList):
    # img_idx = img_idx + 1
    imageList_names = []
    first_BandName = ee.Image(imgList[img_idx]).bandNames().get(0).getInfo() # Used for saving each composite with it's related name "B_Sx" where x = 0, 1 or more"
    imageList_names = imageList_names + [first_BandName]
    print(first_BandName)
    for idx, f in enumerate(features_geopandasList):
      # idx = idx + 267
      pixel_featureCollection = ee.Image(img).sampleRegions(**{
                                                  'collection': geemap.geopandas_to_ee(features_geopandasList[idx]),
                                                  'scale': 10,
                                                  'geometries': True,
                                                  'tileScale': 16
                                                                    })
      task = ee.batch.Export.table.toDrive(**{
                                        'collection': pixel_featureCollection,
                                        'description': year + "_" + str(int(year) + 1) + '_glcm_' + first_BandName.split("_")[1] + "_P" +str(percentile) + '_polygon' + f'{idx}',
                                        'folder': 'DistributionBased_glcmBands_TillageData',
                                        'fileNamePrefix': year + "_" + str(int(year) + 1) + '_glcm_' + first_BandName.split("_")[1] + "_P" +str(percentile) +  '_polygon' + f'{idx}',
                                        'fileFormat': 'CSV'})
      task.start()
      # import time 
      # while task.active():
      #   print('Polling for task (id: {}). Still breathing'.format(task.id))
      #   time.sleep(30)
  return imageList_names



# ///// Function to extract Gray-level Co-occurrence Matrix (GLCM) for each band in the composites  /////
# Input: an imageCollection containing the composites made for a year 
# Output: List of imageCollections with GLCM bands. 
def applyGLCM(coll):
  # Cast image values to a signed 32-bit integer.
  int32Coll = coll.map(lambda img: img.toInt32())
  glcmColl = int32Coll.map(
      lambda img: img.glcmTexture().set("system:time_start", img.date().millis())
      )
  return glcmColl 

# ///// Function to extract percentile values from imageCollections of each year
def applyPercentile(collection, percentileList):
  
  percentileImage = collection.reduce(ee.Reducer.percentile(percentileList))
  percentileImage = percentileImage.set(
      "system:time_start", collection.first().date().millis()
      )
  return percentileImage


# + id="4gbQNHXkicpW" executionInfo={"status": "ok", "timestamp": 1681177129617, "user_tz": 420, "elapsed": 145, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}}
#######################################################################################################
###################      Pre-process Landsat 7 and 8 imageCollections      #################
#######################################################################################################
startYear = 2021
endYear = 2022

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
landSat_7_8 = ee.ImageCollection(L8_2122.merge(L7_2122));

# Apply NDVI mask
landSat_7_8 = landSat_7_8.map(addNDVI)
landSat_7_8 = landSat_7_8.map(maskNDVI)

# Mask Clouds
landSat_7_8 = landSat_7_8.map(cloudMaskL8)

# Mask prercipitation > 3mm two days prior
landSat_7_8 = landSat_7_8.map(MoistMask);

# Add spectral indices to each in the collection as bands
landSat_7_8 = landSat_7_8.map(addIndices);

# + [markdown] id="b_N8FYO9jK5i"
# #### Extract season-based features, using main bands and Gray-level Co-occurence Metrics (GLCMs) values

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 502151, "status": "ok", "timestamp": 1679433498151, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="1OJ1fUM1K-_S" outputId="b13e15f6-6df6-4780-bbae-62de0e3beb51"
#####################################################################
###################      Season-based Features      #################
#####################################################################
# Create season-based composites
# Specify time period
startSeq= 2021
endSeq= 2022
years = list(range(startSeq, endSeq));

# Create season-based composites for each year and put them in a list
yearlyCollectionsList = []
for y in years: 
  yearlyCollectionsList = yearlyCollectionsList + [makeComposite(y, landSat_7_8)]
# Rename bands of each composite in each yearly collection
renamedCollectionList = renameComposites(yearlyCollectionsList)

# Clip each collection to the WSDA field boundaries
clipped_mainBands_CollectionList = list(map(lambda collection: collection.map(lambda img: img.clip(WSDA_featureCol)), renamedCollectionList))

# Extract GLCM metrics
clipped_GLCM_collectionList = list(map(applyGLCM, clipped_mainBands_CollectionList))


# Store each year's collection (composite) containing image pixel values in drive
mainBands_pixel_DataframeList = list(map(mainBand_pixelExtractor, clipped_mainBands_CollectionList))
# glcmBands_pixel_DataframeList = list(map(glcmBand_pixelExtractor, clipped_GLCM_collectionList))


# Display on Map
# Map = geemap.Map()
# Map.setCenter(-117.100, 46.94, 7)
# Map.addLayer(ee.Image(clippedCollectionList[0].toList(clippedCollectionList[0].size()).get(1)), {'bands': ['B4_S1', 'B3_S1', 'B2_S1'], max: 0.5, 'gamma': 2}, 'L8')
# Map

# + id="Asl72S06RTyA"



# + [markdown] id="DUhdHR8xIrUE"
# #### Extract distribution-based (metric-based) features using main bands and Gray-level Co-occurence Metrics (GLCMs) values

# + colab={"base_uri": "https://localhost:8080/"} id="vrRY7E6NLhul" outputId="ccf051b8-9384-42de-f0aa-f2375c8ba0ea" executionInfo={"status": "ok", "timestamp": 1681178083699, "user_tz": 420, "elapsed": 928510, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}}
###########################################################################
###################      Distribution-based Features      #################
###########################################################################

# Create metric composites
# Specify time period
startSeq= 2021
endSeq= 2022
years = list(range(startSeq, endSeq));

# Create a list of lists of imageCollections. Each year would have n number of imageCollection corresponding to the time periods specified 
# for creating metric composites. 
yearlyCollectionsList = []
for y in years: 
  yearlyCollectionsList = yearlyCollectionsList + [groupImages(y, landSat_7_8)]  # 'yearlyCollectionsList' is a Python list
yearlyCollectionsList[0][0]

# Clip each collection to the WSDA field boundaries
clipped_mainBands_CollectionList = list(map(lambda collList: list(map(lambda collection: ee.ImageCollection(collection).map(lambda img: img.clip(WSDA_featureCol)), collList)), yearlyCollectionsList))

# Extract GLCM metrics
clipped_GLCM_collectionList = list(map(lambda collList: list(map(applyGLCM, collList)), clipped_mainBands_CollectionList))

# Compute percentiles and extract pixel values and store them on google drive
percentiles = [25]
for p in percentiles:
  mainBands_percentile_collectionList = list(map(lambda collList: list(map(lambda collection: applyPercentile(collection, [p]), collList)), clipped_mainBands_CollectionList))
  # glcmBands_percentile_collectionList = list(map(lambda collList: list(map(lambda collection: applyPercentile(collection, [p]), collList)), clipped_GLCM_collectionList))
  list(map(lambda imgList: metricBased_mainBand_pixelExtractor(imgList, p), mainBands_percentile_collectionList))
  # list(map(lambda imgList: metricBased_glcmBand_pixelExtractor(imgList, p), glcmBands_percentile_collectionList))

