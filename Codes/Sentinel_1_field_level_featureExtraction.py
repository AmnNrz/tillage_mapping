# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="whtD--m_FObX" executionInfo={"status": "ok", "timestamp": 1688412727824, "user_tz": 420, "elapsed": 38684, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}} outputId="8e3adb79-c856-42fa-f504-ec72a892eb08"
# Initialize GEE python API
import ee
# Trigger the authentication flow.
ee.Authenticate()
# Initialize the library.
ee.Initialize()

# Install geemap
# !pip install geemap
# !pip install geopandas
import ee
import geemap
import numpy as np
import geopandas as gpd
import pandas as pd

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 13944, "status": "ok", "timestamp": 1688412743734, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="aYGl9T8OQZrA" outputId="5376bef5-3a0a-4fea-c809-f778c3cd3535"
# Mount google drive
from google.colab import drive
drive.mount('/content/drive')

# + [markdown] id="zNHfo5P4FUfQ"
# # Download raw and derived indices data for residue and crop-type classification from GEE imagery data:

# + [markdown] id="MWRiGjZRHCOf"
# #### Imports

# + colab={"base_uri": "https://localhost:8080/", "height": 641, "referenced_widgets": ["52cb6442a99547c59ea159281295741b", "59669cc811c6458d9cefe1752a7325a6", "ebe02718fba14ea0b58da9f0de86b49e", "fac87ffbdd304d629061809d1c701395", "7de864bf42bb481c967b48ab7912d13a", "dea6b11115f1487c995608ade6ca9266", "53dafeee403543319774a6f332b8ec4d", "8175c0c9b57b488a9f5ccce9d04208fe", "f1bad9251fc14047a1a66d688318cdce", "774995f3013c46b1afbff3862560397a", "fac1c183c7684f209d4a291f30018a96", "3d4f2a70bddf403f89e2ca00c7c92de0", "29e13e4f7504468aba1ed2cb323589f3", "1e6a7be24e644e0fae29f03da8a8fdfc", "da6a9e19b4a64d7496f5cab78b56b460", "a804746aa6114bbc969c0c5fca66223b", "738b2d638ed4469381364a1e1064ac24", "926581edb40c44ca9b4cd20008f218fd", "ab7f0876ed944b1881b7a35231e83341", "10e9cfd7302a42718dce4ccc7285c37a", "9bd796e683a0431f8716a4a0cf625b18", "17b02fb1266243929b5d6b55db95aea3", "82153fdcc14d4ce89a8ca87a1750bbe2", "3fa90ba502d94ca4babc39da0230b0ca", "30621a4b62a74b52ba586d59463f7da3", "100a878e517046729c382c8e58faa525", "9bd849c7e77e4b93b07431b8a91f62c4", "fa676db198214e69906b24e28571497d", "0e0af514c02a4f56943cdb883f71b189", "912a73193fed4fe280c847136bff56f7", "e6c62142dac64203a86af99a2f3cae6e", "58aa26f32ce8490d81e08468b4d1327a", "e7e3957bcec44aada035d7e42dc5fa42", "89ee842a966c46f7a620895ce652ef9a", "a257dcdd88f24a44b1e004b1ff656052", "34f573c5565f45288739d65441d10523", "8a63af386954457d9e3021024773699f", "9e0086aa0ede4cd5b6e88db2bd484029", "3f6c0f3cb6cb4fa6a265cfeb368a2545", "912a2188639d482494c827dd3f554311", "28939bfec52e46b2a2bd04db9580fb7e", "b624a7087a3f4e5c83040ffbc6145a9f", "4b75498d87ad438da4b1dea329743564", "cdaca02d94f645e6ad27d8d8d99f67e0", "76bc3161d8fa4ce3820edd1d275a77f7", "012ae2a2a792443bb43aff74cf22732f", "4823a32bb4cf4a7794b86d17d1248a43", "db2cfe846ffc4a5d9c5342f096a2687d", "3c129659836c4fc5959b9302fde5aa2f", "85c3642e870e46bba96e78a401e50ca0", "59b6b792e264475d8a841d158681fe79", "fd7ce615926a4fa79b42e5c79332c551", "b9c90b156b0d432895d4863cf34cb59d", "acdf1fe4c0764a529b4593aed053d542", "d6efc2343ea24c28a68d83fadef94eab", "b021c14f02f748019653987d2ce97135", "1c8dfd670f464440a1ba7d34cf3e5efa", "137b2bb41df741d4b327c51dd7442c95", "c7fa8d994c58415b988df32cc1c6419f", "c753daa97ca543b586283b3bbedb68dc", "21d1848daa1f4d8fa948cda8b92fe032", "00380c5776a648c7b9836036aa174edd", "e29d49c325054930a7d438e316ae41fb", "f0918d1a26c64fbda9c7863248b44827", "3166a700c4be401193edc18fb0afae2b", "02f6ac81f63b4725bfaa980540513f1c", "c4265875b61546aba1e7b7cd2a75e199", "17e1f2fb53604a7f874fa2c02cc9d225", "edc6edd7105c4c38a33c2c09c304414a", "31982b6dabdc40cda6585ef49ec0c49e", "3bc69084219e4fb4bda3c88d4d0c306c", "e139dc7c63a243f9817e136994a27c89", "3dbd8f4314c9497f89413b70cd1aaf1c", "5ecd4258181c49399a43ff930096bf5e", "1e280c1024c9416b9e5426c90019e994", "7306df28c8ad48da829e6349105c64f9", "78c4c0dfd9304b1fa4b75d57caa56100", "54660585cb19427d818a57f2233d038d", "923ae2d148e042c9a37e308f686dc72e", "8c7304eb4c6a485aa86c0d4bc292db5e", "a4bb716b05b845f88c1f5cdb1f498856", "7de2420eff1941e59b4c06030372a570", "0a4743188b1043e4a236aec6b53fca69", "ca797b78785c496a9de05503f6985174", "68ce10ef9ec141b18e497dc6605a0b1d", "3a39d8a978c646ab8d42be203db39796", "eb9549525b6341f1891d1f94d8499d23", "5f97ae082d42430bba4305d2bd602bcb", "fe60dc25fe3f4a279c5ee4c9f193448a", "5033310ceeb34931a359df6411e6908c", "51a822b88b3e44719552ebc3978a1e7c", "9bd54ba4c6d9432d9d114cba6c3b82cc", "15e3910a652f423095ee5485132a14e8", "c1936e2fc8e94fec8577b106afa3dab6", "f0ca74eb29d04c119364367e0433d9d4", "cca25d57684d46f48b90e2fffb07c0c1", "70c3b9bc46414fa389932f29cb5368e5", "23a3cedb41164187aabf9e086187fa93", "2f2255cf17744c1398dbfdefdb70b641", "f9944ac3fda3413d85cf12eb1d27005f"]} executionInfo={"elapsed": 494, "status": "ok", "timestamp": 1688413064980, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="RkSJ3M7GG_pD" outputId="b76b4065-11ab-47f5-86b4-94a71db0381c"
######## imports #########
# Import WSDA polygons of surveyed fields
# consider a polygon that covers the study area (Whitman & Columbia counties)
geometry = ee.Geometry.Polygon(
        [[[-118.61039904725511, 47.40441980731236],
          [-118.61039904725511, 45.934467488469],
          [-116.80864123475511, 45.934467488469],
          [-116.80864123475511, 47.40441980731236]]], None, False)

geometry3 = ee.Geometry.Polygon(
        [[[-117.2517791873828,46.663694511306254],
        [-117.23006402319822,46.663694511306254],
        [-117.23006402319822,46.67941954646265],
        [-117.2517791873828,46.67941954646265],
        [-117.2517791873828,46.663694511306254]]], None, False)


geometry2 = ee.Geometry.Point([-117.10053796709163, 46.94957951590986]),
WSDA_featureCol = ee.FeatureCollection("projects/ee-bio-ag-tillage/assets/2021_2022_pols")

# #import USGS Landsat 8 Level 2, Collection 2, Tier 1
# L8T1 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

# Import Sentinel 1
# Filter the Sentinel-1 collection based on AOI, date range, polarization, and instrument mode
collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
  .filterBounds(geometry) \
  .filterDate('2020-01-01', '2021-12-30') \
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
  .filter(ee.Filter.eq('instrumentMode', 'IW')) \
  .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));

# Function to convert linear to dB scale for Sentinel-1 data
def toDbS1(image):
  dB = ee.Image(10.0).multiply(image.log10()).toFloat();

  # Rename the bands for VV and VH
  bands = image.bandNames();
  newBands = bands.map(lambda band: ee.String(band).cat('_dB'));

  # Add dB bands and rename them
  imageWithDb = image.addBands(dB);
  renamedImage = imageWithDb.select(bands, newBands);

  return renamedImage;




# Apply preprocessing and visualization
processedCollection = collection \
  .map(toDbS1) \
  .map(lambda image: image.select(['VV_dB', 'VH_dB']))

# // Add the filtered collection to the map
# Map.addLayer(filtered, {bands: ['VV_dB', 'VH_dB'], min: -20, max: 0}, 'Sentinel-1');

# // Center the map display on a specific location
# Map.centerObject(filtered, 10);

# Display on Map
Map = geemap.Map()
Map.setCenter(-117.100, 46.94, 7)
Map.addLayer(processedCollection, {'bands': ['VV_dB', 'VH_dB'], min: -20, max: 0}, 'Sentinel-1')
Map

# + [markdown] id="dl5KSrInfIGI"
# #### Functions

# + id="QaaLjXabmhWA" executionInfo={"status": "ok", "timestamp": 1688413182462, "user_tz": 420, "elapsed": 116, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}}
#######################     Functions     ######################

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

# ///// Function to add day of year (DOY) to each image as a band /////
def addDOY(img):
  doy = img.date().getRelative('day', 'year');
  doyBand = ee.Image.constant(doy).uint16().rename('doy')
  doyBand
  return img.addBands(doyBand)

# # ///// Function to make metric-based imageCollections /////
# # This groups images in a year and returns a list of imageCollections.
# def groupImages(year, orgCollection):
# # This groups images and rename bands
#   bands = ['B', 'G', 'R', 'NIR', 'SWIR1', 'SWIR2', 'evi', 'gcvi', 'ndvi', 'sndvi', 'ndti', 'ndi5', 'ndi7', 'crc', 'sti', 'doy'];
#   new_bandS0 = ['B_S0', 'G_S0', 'R_S0', 'NIR_S0', 'SWIR1_S0', 'SWIR2_S0', 'evi_S0', 'gcvi_S0', 'ndvi_S0', 'sndvi_S0', 'ndti_S0', 'ndi5_S0', 'ndi7_S0', 'crc_S0', 'sti_S0', 'doy_S0'];
#   new_bandS1 = ['B_S1', 'G_S1', 'R_S1', 'NIR_S1', 'SWIR1_S1', 'SWIR2_S1', 'evi_S1', 'gcvi_S1', 'ndvi_S1', 'sndvi_S1', 'ndti_S1', 'ndi5_S1', 'ndi7_S1', 'crc_S1', 'sti_S1', 'doy_S1'];
#   new_bandS2 = ['B_S2', 'G_S2', 'R_S2', 'NIR_S2', 'SWIR1_S2', 'SWIR2_S2', 'evi_S2', 'gcvi_S2', 'ndvi_S2', 'sndvi_S2', 'ndti_S2', 'ndi5_S2', 'ndi7_S2', 'crc_S2', 'sti_S2', 'doy_S2'];
#   new_bandS3 = ['B_S3', 'G_S3', 'R_S3', 'NIR_S3', 'SWIR1_S3', 'SWIR2_S3', 'evi_S3', 'gcvi_S3', 'ndvi_S3', 'sndvi_S3', 'ndti_S3', 'ndi5_S3', 'ndi7_S3', 'crc_S3', 'sti_S3', 'doy_S3'];

#   year = ee.Number(year)
#   collection_1 = orgCollection.filterDate(
#       ee.Date.fromYMD(year, 9, 1),
#       ee.Date.fromYMD(year, 12, 30)
#     ).filterBounds(geometry)\
#     .map(addDOY)\
#     .map(lambda img: img.select(bands).rename(new_bandS0))



#     # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(year, 9, 1).millis()))

#   collection_2 = orgCollection\
#     .filterDate(
#       ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1),
#       ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 30)
#     ).filterBounds(geometry)\
#     .map(addDOY)\
#     .map(lambda img: img.select(bands).rename(new_bandS1))

#     # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 1, 1).millis()))

#   collection_3 = orgCollection\
#     .filterDate(
#       ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1),
#       ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 8, 30)
#     ).filterBounds(geometry)\
#     .map(addDOY)\
#     .map(lambda img: img.select(bands).rename(new_bandS2))

#     # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 5, 1).millis()))

#   collection_4 = orgCollection\
#     .filterDate(
#       ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1),
#       ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 12, 30)
#     ).filterBounds(geometry)\
#     .map(addDOY)\
#     .map(lambda img: img.select(bands).rename(new_bandS3))

#     # .map(lambda img: img.set('system:time_start', ee.Date.fromYMD(ee.Number(year).add(ee.Number(1)), 9, 1).millis()))

#   # Return a list of imageCollections

#   return [collection_1, collection_2, collection_3, collection_4]

# ///// Function to rename the bands of each composite in the imageCollections associated with each year /////
def renameComposites(collectionList):
  renamedCollectionList = []
  for i in range(len(collectionList)):
    ith_Collection = collectionList[i]
    Comp_S0 = ith_Collection.toList(ith_Collection.size()).get(0);
    Comp_S1 = ith_Collection.toList(ith_Collection.size()).get(1);
    Comp_S2 = ith_Collection.toList(ith_Collection.size()).get(2);
    Comp_S3 = ith_Collection.toList(ith_Collection.size()).get(3);

    bands = ['VV_dB', 'VH_dB'];
    new_bandS0 = ['VV_dB_S0', 'VH_dB_S0'];
    new_bandS1 = ['VV_dB_S1', 'VH_dB_S1'];
    new_bandS2 = ['VV_dB_S2', 'VH_dB_S2'];
    new_bandS3 = ['VV_dB_S3', 'VH_dB_S3'];

    composite_S0 = ee.Image(Comp_S0).select(bands).rename(new_bandS0)
    composite_S1 = ee.Image(Comp_S1).select(bands).rename(new_bandS1)
    composite_S2 = ee.Image(Comp_S2).select(bands).rename(new_bandS2)
    composite_S3 = ee.Image(Comp_S3).select(bands).rename(new_bandS3)

    renamedCollection = ee.ImageCollection.fromImages([composite_S0, composite_S1, composite_S2, composite_S3]);
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

# ///// Function to reduce each image in a collection (with different band names for each image) to
# a mean value (mean value over each field geometry) /////
def collectionReducer(imgcollection):
  imageList = eeList_to_pyList(imgcollection.toList(imgcollection.size()))
  return list(map(lambda img:ee.Image(img).reduceRegions(**{
                                                  'collection':WSDA_featureCol,
                                                  'reducer':ee.Reducer.mean(),
                                                  'scale': 1000

                                                }), imageList))

# ///// Function to reduce each percentile image (with different band names for each image) to
# a mean value (mean value over each field geometry) /////
def percentile_imageReducer(imageList):
  return list(map(lambda img: ee.Image(img).reduceRegions(**{
      'reducer': ee.Reducer.mean(),
      'collection': WSDA_featureCol,
      'scale': 1000,
      'tileScale': 16
  }), imageList))

# ///// Function to create pandas dataframes from geographically (by field) reduced featureCollections  /////
# Arguments: 1) List of lists of featureCollections:
#                              [[y1_f0, y1_f1, y1_f2, y1_f3], [y2_f0, y2_f1, y2_f2, y2_f3], ..., [yn_f0, yn_f1, yn_f2, yn_f3]]
#                              y1_f0 : season 1 (or time period 1) of year 1 reduced composite
# Output: Lists of dataframes. Each dataframe is the derived data for each year.
def eefeatureColl_to_Pandas(yearlyList, bandNames):
  dataList = []   # This list is going to contain dataframes for each year data
  for i in range(len(yearlyList)):
    year_i = pyList_to_eeList(yearlyList[i])
    important_columns = ['pointID'] + bandNames

    df_yi = pd.DataFrame([])
    for j in range(year_i.length().getInfo()):
      f_j = year_i.get(j)  # Jth featureCollection (reduced composite data)
      df_j = geemap.ee_to_pandas(ee.FeatureCollection(f_j))  # Convert featureCollection to pandas dataframe
      df_j = df_j[df_j.columns[(df_j.columns).isin(important_columns)]]   # Pick needed columns
      df_yi = pd.concat([df_yi, df_j], axis=1)
    df_yi = df_yi.loc[:,~df_yi.columns.duplicated()]   # Drop repeated 'pointID' columns

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

# # ///// Function to create metric composites /////
# percentiles = [5, 25, 50, 75, 100]
# def metricComposite(allyears_CollectionList, years):
#   for p in percentiles:
#     Bands_percentile_collectionList = list(map(lambda collList: list(map(lambda collection: \
#                                     collection.reduce(ee.Reducer.percentile([p])), collList)), allyears_CollectionList))
#     for y in range(len(years)):
#       subtracted_collectionList = list(map(lambda org_collection, img_pth: org_collection.map(lambda img: img.subtract(img_pth)),\
#                                      clipped_mainBands_CollectionList[y], mainBands_percentile_collectionList[y]))
#       absolute_difference = list(map(lambda collection: collection.map(lambda img: img.abs()), subtracted_collectionList))

#       # Choose min absolute differences to find the
#       # min_


# + [markdown] id="Xi8j9i9nSiW7"
# #### Extract season-based features, using main bands and Gray-level Co-occurence Metrics (GLCMs) values

# + colab={"base_uri": "https://localhost:8080/", "height": 193, "output_embedded_package_id": "1BBaUeQO9vK_ScCWNXM8L_VXabq-CetUl"} id="wMm7TTN-R7a4" executionInfo={"status": "ok", "timestamp": 1688413393077, "user_tz": 420, "elapsed": 1955, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}} outputId="5c8c9464-8a12-40f6-dba7-0f1ea11365aa"
processedCollection

# + colab={"base_uri": "https://localhost:8080/", "height": 387} executionInfo={"elapsed": 128, "status": "error", "timestamp": 1688413194958, "user": {"displayName": "Amin Norouzi Kandelati", "userId": "14243952765795155999"}, "user_tz": 420} id="1OJ1fUM1K-_S" outputId="3ecbe32e-c87e-458e-ad87-b9e376bc9a26"
#####################################################################
###################      Season-based Features      #################
#####################################################################
startYear = 2021
endYear = 2022

S_1_collection = processedCollection\
  .filter(ee.Filter.calendarRange(startYear, endYear, 'year'))\
  .map(lambda img: img.set('year', img.date().get('year')))\
  .map(lambda img: img.clip(geometry))

# Create season-based composites
# Specify time period
startSeq= 2021
endSeq= 2022
years = list(range(startSeq, endSeq));

# Create season-based composites for each year and put them in a list
yearlyCollectionsList = []
for y in years:
  yearlyCollectionsList = yearlyCollectionsList + [makeComposite(y, S_1_collection)]
# Rename bands of each composite in each yearly collection
renamedCollectionList = renameComposites(yearlyCollectionsList)

# Clip each collection to the WSDA field boundaries
clipped_mainBands_CollectionList = list(map(lambda collection: collection.map(lambda img: img.clip(WSDA_featureCol)), renamedCollectionList))
clipped_mainBands_CollectionList[0]
# Extract GLCM metrics
clipped_GLCM_collectionList = list(map(applyGLCM, clipped_mainBands_CollectionList))

clipped_mainBands_CollectionList[0]

# Extract band names to use in our dataframes
# The bands are identical for all years so we use the first year imageCollection, [0]
# Main bands:
imageList = eeList_to_pyList(clipped_mainBands_CollectionList[0].toList(clipped_mainBands_CollectionList[0].size()))
nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(), imageList))
mainBands = [name for sublist in nameLists for name in sublist]

# GLCM bands:
imageList = eeList_to_pyList(clipped_GLCM_collectionList[0].toList(clipped_GLCM_collectionList[0].size()))
nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(), imageList))
glcmBands = [name for sublist in nameLists for name in sublist]

# Reduce each image in the imageCollections (with main bands) to mean value over each field (for each year)
reducedList_mainBands = list(map(collectionReducer, clipped_mainBands_CollectionList))   # This will produce a list of lists containing reduced featureCollections

# Reduce each image in the imageCollections (with GLCM bands) to mean value over each field (for each year)
reducedList_glcmBands = list(map(collectionReducer, clipped_GLCM_collectionList))

# Convert each year's composites to a single dataframe and put all the dataframes in a list
seasonBased_dataframeList_mainBands = eefeatureColl_to_Pandas(reducedList_mainBands, mainBands)
seasonBased_dataframeList_glcm = eefeatureColl_to_Pandas(reducedList_glcmBands, glcmBands)
print(seasonBased_dataframeList_mainBands[0].shape)
print(seasonBased_dataframeList_glcm[0].shape)

# // Add the filtered collection to the map
# Map.addLayer(filtered, {bands: ['VV_dB', 'VH_dB'], min: -20, max: 0}, 'Sentinel-1');

# // Center the map display on a specific location
# Map.centerObject(filtered, 10);

# Display on Map
# Map = geemap.Map()
# Map.setCenter(-117.100, 46.94, 7)
# Map.addLayer(ee.Image(clippedCollectionList[0].toList(clippedCollectionList[0].size()).get(1)), {'bands': ['B4_S1', 'B3_S1', 'B2_S1'], max: 0.5, 'gamma': 2}, 'L8')
# Map

# + [markdown] id="DUhdHR8xIrUE"
# #### Extract distribution-based (metric-based) features using main bands and Gray-level Co-occurence Metrics (GLCMs) values

# + colab={"background_save": true} id="vrRY7E6NLhul"
from functools import reduce
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

# # Compute percentiles
percentiles = [5, 25, 50, 75, 100]
mainBands_percentile_collectionList = list(map(lambda collList: list(map(lambda collection: collection.reduce(ee.Reducer.percentile(percentiles)), collList)), clipped_mainBands_CollectionList))
glcmBands_percentile_collectionList = list(map(lambda collList: list(map(lambda collection: collection.reduce(ee.Reducer.percentile(percentiles)), collList)), clipped_GLCM_collectionList))

# Reduce each image in the imageCollections (with main bands) to mean value over each field (for each year)
reducedList_mainBands = list(map(percentile_imageReducer, mainBands_percentile_collectionList))   # This will produce a list of lists containing reduced featureCollections

# Reduce each image in the imageCollections (with GLCM bands) to mean value over each field (for each year)
reducedList_glcmBands = list(map(percentile_imageReducer, glcmBands_percentile_collectionList))

# Extract band names to use in our dataframes
# The bands are identical for all years so we use the first year imageCollection, [0]
# Main bands:
# imageList = eeList_to_pyList(clipped_mainBands_CollectionList[0])
nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(), mainBands_percentile_collectionList[0]))
mainBands = [name for sublist in nameLists for name in sublist]

# GLCM bands:
# imageList = eeList_to_pyList(clipped_GLCM_collectionList[0].toList(clipped_GLCM_collectionList[0].size()))
nameLists = list(map(lambda img: ee.Image(img).bandNames().getInfo(), glcmBands_percentile_collectionList[0]))
glcmBands = [name for sublist in nameLists for name in sublist]

# Convert each year's composites to a single dataframe and put all the dataframes in a list
# The dataframes include pointID (first column), mainbands, derived indices and the corresponding GLCM metrics
metricBased_dataframeList_mainBands = eefeatureColl_to_Pandas(reducedList_mainBands, mainBands)
metricBased_dataframeList_glcm = eefeatureColl_to_Pandas(reducedList_glcmBands, glcmBands)

# + [markdown] id="YkqsHChCZNTy"
# # Estimate crop-residue type using an ML model

# + [markdown] id="CzX3clorromO"
# #### Prepare data

# + id="rg5ScAFczWH5"
# Set the directory and import
import sys
sys.path.insert(0, "/content/drive/MyDrive/PhD/Bio_AgTillage/01. Codes/pipeline_tillageClassifier")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import NASA_core as nc

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 15699, "status": "ok", "timestamp": 1676696363000, "user": {"displayName": "amin norouzi", "userId": "02798455485104728835"}, "user_tz": 480} id="IPTNg1FbRLGg" outputId="94d5af90-285e-48e4-d6e6-1f4bccdbfba8"
# colnames = ['pointID', 'LastSurveyDate', 'CropType', 'NDVI', 'EVI', 'blue', 'green', 'red', 'ni', 'swi1',
#             'swi2', 'system_start_time']
# directory = "/content/drive/MyDrive/PhD/Bio_AgTillage/01. Codes/pipeline_tillageClassifier/CropClassification Data/"
# l8raw_2018_2019 = pd.read_csv(directory + "L8_T1C2L2_timeSeries_2018-08-01_2019-07-31.csv")
# l8raw_2019_2020 = pd.read_csv(directory + "L8_T1C2L2_timeSeries_2019-08-01_2020-07-31.csv")
# l8raw_2020_2021 = pd.read_csv(directory + "L8_T1C2L2_timeSeries_2020-08-01_2021-07-31.csv")
# l7raw_2018_2019 = pd.read_csv(directory + "L7_T1C2L2_timeSeries_2018-08-01_2019-07-31.csv")
# l7raw_2019_2020 = pd.read_csv(directory + "L7_T1C2L2_timeSeries_2019-08-01_2020-07-31.csv")
# l7raw_2020_2021 = pd.read_csv(directory + "L7_T1C2L2_timeSeries_2020-08-01_2021-07-31.csv")

# l8raw_2018_2019.columns = colnames
# l8raw_2019_2020.columns = colnames
# l8raw_2020_2021.columns = colnames
# l7raw_2018_2019.columns = colnames
# l7raw_2019_2020.columns = colnames
# l7raw_2020_2021.columns = colnames

# print(l8raw_2018_2019.columns)
# print(l7raw_2018_2019.columns)

# l87raw_2018_2019 = pd.concat([l8raw_2018_2019, l7raw_2018_2019])
# l87raw_2019_2020 = pd.concat([l8raw_2019_2020, l7raw_2019_2020])
# l87raw_2020_2021 = pd.concat([l8raw_2020_2021, l7raw_2020_2021])

# l87raw_2018_2019 = l87raw_2018_2019[~l87raw_2018_2019.isnull()["NDVI"]]
# l87raw_2019_2020 = l87raw_2019_2020[~l87raw_2019_2020.isnull()["NDVI"]]
# l87raw_2020_2021 = l87raw_2020_2021[~l87raw_2020_2021.isnull()["NDVI"]]

# + id="NAjg3pFwrB8J"
# # Function to convert system_start_time to human readable format
# def add_human_start_time_by_system_start_time(HDF):
#     """Returns human readable time (conversion of system_start_time)
#     Arguments
#     ---------
#     HDF : dataframe
#     Returns
#     -------
#     HDF : dataframe
#         the same dataframe with added column of human readable time.
#     """
#     HDF.system_start_time = HDF.system_start_time / 1000
#     time_array = HDF["system_start_time"].values.copy()
#     human_time_array = [time.strftime('%Y-%m-%d', time.localtime(x)) for x in time_array]
#     HDF["human_system_start_time"] = human_time_array

#     if type(HDF["human_system_start_time"]==str):
#         HDF['human_system_start_time'] = pd.to_datetime(HDF['human_system_start_time'])

#     """
#     Lets do this to go back to the original number:
#     I added this when I was working on Colab on March 30, 2022.
#     Keep an eye on it and see if we have ever used "system_start_time"
#     again. If we do, how we use it; i.e. do we need to get rid of the
#     following line or not.
#     """
#     HDF.system_start_time = HDF.system_start_time * 1000
#     return(HDF)

# # Convert
# l87raw_2018_2019 = add_human_start_time_by_system_start_time(l87raw_2018_2019)
# l87raw_2019_2020 = add_human_start_time_by_system_start_time(l87raw_2019_2020)
# l87raw_2020_2021 = add_human_start_time_by_system_start_time(l87raw_2020_2021)

# merged_df = pd.concat([l87raw_2018_2019, l87raw_2019_2020, l87raw_2020_2021])
# merged_df

# + id="W0Me9sD6KD1n"



# + [markdown] id="DVlIa3KVslVN"
# ##### Remove Outliers

# + colab={"base_uri": "https://localhost:8080/"} id="CR4yKQf1sqj5" outputId="9ca16bf9-4084-4fc3-ac16-0f3faed83d94"
# merged_df["pointID"] = merged_df["pointID"].astype(str)
# IDs = np.sort(merged_df["pointID"].unique())
# indeks = "NDVI"

# no_outlier_df = pd.DataFrame(data = None,
#                          index = np.arange(merged_df.shape[0]),
#                          columns = merged_df.columns)
# counter = 0
# row_pointer = 0
# for a_poly in IDs:
#     if (counter % 1000 == 0):
#         print ("counter is [{:.0f}].".format(counter))
#     curr_field = merged_df[merged_df["pointID"]==a_poly].copy()
#     # small fields may have nothing in them!
#     if curr_field.shape[0] > 2:
#         ##************************************************
#         #
#         #    Set negative index values to zero.
#         #
#         ##************************************************

#         # curr_field.loc[curr_field[indeks] < 0 , indeks] = 0
#         no_Outlier_TS = nc.interpolate_outliers_EVI_NDVI(outlier_input = curr_field, given_col = indeks)
#         no_Outlier_TS.loc[no_Outlier_TS[indeks] < 0 , indeks] = 0

#         if len(no_Outlier_TS) > 0:
#             no_outlier_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
#             counter += 1
#             row_pointer += curr_field.shape[0]

# # Sanity check. Will neved occur. At least should not!
# no_outlier_df.drop_duplicates(inplace=True)

# + [markdown] id="BSR7kkL3sy5u"
# ##### Remove Jumps

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 2153186, "status": "ok", "timestamp": 1676705481278, "user": {"displayName": "amin norouzi", "userId": "02798455485104728835"}, "user_tz": 480} id="7JrWsVe-s2VS" outputId="80539332-5f5c-467b-d527-9877eea50a24"

# noJump_df = pd.DataFrame(data = None,
#                          index = np.arange(no_outlier_df.shape[0]),
#                          columns = no_outlier_df.columns)
# counter = 0
# row_pointer = 0

# for a_poly in IDs:
#     if (counter % 1000 == 0):
#         print ("counter is [{:.0f}].".format(counter))
#     curr_field = no_outlier_df[no_outlier_df["pointID"]==a_poly].copy()

#     ################################################################
#     # Sort by DoY (sanitary check)
#     curr_field.sort_values(by=['human_system_start_time'], inplace=True)
#     curr_field.reset_index(drop=True, inplace=True)

#     ################################################################

#     no_Outlier_TS = nc.correct_big_jumps_1DaySeries_JFD(dataTMS_jumpie = curr_field,
#                                                         give_col = indeks,
#                                                         maxjump_perDay = 0.018)

#     noJump_df[row_pointer: row_pointer + curr_field.shape[0]] = no_Outlier_TS.values
#     counter += 1
#     row_pointer += curr_field.shape[0]

# noJump_df['human_system_start_time'] = pd.to_datetime(noJump_df['human_system_start_time'])

# # Sanity check. Will neved occur. At least should not!
# print ("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape))
# noJump_df.drop_duplicates(inplace=True)
# print ("Shape of noJump_df before dropping duplicates is {}.".format(noJump_df.shape))

# + colab={"base_uri": "https://localhost:8080/", "height": 208} executionInfo={"elapsed": 495, "status": "ok", "timestamp": 1676705482189, "user": {"displayName": "amin norouzi", "userId": "02798455485104728835"}, "user_tz": 480} id="gSPX9EBP67JN" outputId="813fb706-8c94-4040-bf61-6cae65d7f18d"
# # Plot a field timeseries
# field_1 = noJump_df[noJump_df["pointID"] == 2]

# # Plot
# fig, ax = plt.subplots(1, 1, figsize=(20, 3),
#                        sharex='col', sharey='row',
#                        # sharex=True, sharey=True,
#                        gridspec_kw={'hspace': 0.2, 'wspace': .05})
# ax.grid(True)

# ax.scatter(field_1['human_system_start_time'], field_1["NDVI"], s=40, c='#d62728')
# ax.plot(field_1['human_system_start_time'], field_1["NDVI"],
#         linestyle='-',  linewidth=3.5, color="#d62728", alpha=0.8,
#         label="raw NDVI")
# plt.ylim([0, 1])
# ax.legend(loc="lower right")

# + id="7woh0FRuHa1x"
# noJump_df.to_csv("noJump_df.csv")

# + id="HQpALpKOHvsx"
# noJump_df = pd.read_csv("/content/drive/MyDrive/PhD/Bio_AgTillage/01. Codes/pipeline_tillageClassifier/noJump_df.csv")

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 163, "status": "ok", "timestamp": 1676954898731, "user": {"displayName": "amin norouzi", "userId": "02798455485104728835"}, "user_tz": 480} id="rGJfduanSJ3L" outputId="72ab13e2-8f0c-4f64-9fd1-5640d05b5a83"
# noJump_df.columns

# + id="dZ4W1z_pR2s9"
# noJump_df.rename(columns = {'Unnamed: 0':'index'}, inplace = True)

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1882, "status": "ok", "timestamp": 1677018712381, "user": {"displayName": "amin norouzi", "userId": "02798455485104728835"}, "user_tz": 480} id="ECX0sZKVQsou" outputId="e77c1bf7-d5f8-4df8-a548-c5e6edc42d2d"
# noJump_df["human_system_start_time"] = pd.to_datetime(noJump_df["human_system_start_time"])


# window_size = 10
# indeks = "red"
# reg_cols = ['pointID', 'human_system_start_time', "indeks"]
# IDs = np.sort(noJump_df["pointID"].unique())
# startYear = noJump_df["human_system_start_time"].dt.year.max()
# endYear = noJump_df["human_system_start_time"].dt.year.min()
# numberOfdays = (startYear - endYear + 1)*366

# nsteps = int(np.ceil(numberOfdays / window_size))

# nrows = nsteps * len(IDs)
# print('st_yr is {}.'.format(startYear))
# print('end_yr is {}.'.format(endYear))
# print('nrows is {}.'.format(nrows))


# + id="aT7TJM8PQ6lI"
# regular_df = pd.DataFrame(data = None,
#                          index = np.arange(nrows),
#                          columns = reg_cols)

# counter = 0
# row_pointer = 0

# for a_poly in IDs:
#     if (counter % 1000 == 0):
#         print ("counter is [{:.0f}].".format(counter))
#     curr_field = noJump_df[noJump_df["pointID"]==a_poly].copy()
#     ################################################################
#     # Sort by date (sanitary check)
#     curr_field.sort_values(by=['human_system_start_time'], inplace=True)
#     curr_field.reset_index(drop=True, inplace=True)

#     ################################################################
#     regularized_TS = nc.regularize_a_field(a_df = curr_field, \
#                                            V_idks = indeks, \
#                                            interval_size = window_size,\
#                                            start_year = startYear, \
#                                            end_year = endYear)

#     regularized_TS = nc.fill_theGap_linearLine(a_regularized_TS = regularized_TS, V_idx = indeks)
#     # if (counter == 0):
#     #     print ("regular_df columns:",     regular_df.columns)
#     #     print ("regularized_TS.columns", regularized_TS.columns)

#     ################################################################
#     # row_pointer = no_steps * counter

#     """
#        The reason for the following line is that we assume all years are 366 days!
#        so, the actual thing might be smaller!
#     """
#     # why this should not work?: It may leave some empty rows in regular_df
#     # but we drop them at the end.
#     print(regularized_TS.shape[0])
#     print(regular_df[row_pointer : (row_pointer+regularized_TS.shape[0])].shape)
#     regular_df[row_pointer : (row_pointer+regularized_TS.shape[0])] = regularized_TS.values
#     row_pointer += regularized_TS.shape[0]

#     # right_pointer = row_pointer + min(no_steps, regularized_TS.shape[0])
#     # print('right_pointer - row_pointer + 1 is {}!'.format(right_pointer - row_pointer + 1))
#     # print('len(regularized_TS.values) is {}!'.format(len(regularized_TS.values)))
#     # try:
#     #     ### I do not know why the hell the following did not work for training set!
#     #     ### So, I converted this to try-except statement! hopefully, this will
#     #     ### work, at least as temporary remedy! Why it worked well with 2008-2021 but not 2013-2015
#     #     regular_df[row_pointer: right_pointer] = regularized_TS.values
#     # except:
#     #     regular_df[row_pointer: right_pointer+1] = regularized_TS.values
#     counter += 1

# regular_df['human_system_start_time'] = pd.to_datetime(regular_df['human_system_start_time'])
# regular_df.drop_duplicates(inplace=True)
# regular_df.dropna(inplace=True)

# # Sanity Check
# regular_df.sort_values(by=["pointID", 'human_system_start_time'], inplace=True)
# regular_df.reset_index(drop=True, inplace=True)

# + id="4zgpySnUQ808"
# regular_df.to_csv("regular_df.csv")

# + id="ZNjlQ6dnuZ5m"
# regular_df = pd.read_csv('/content/drive/MyDrive/PhD/Bio_AgTillage/01. Codes/pipeline_tillageClassifier/regular_df.csv',
                        #  index_col=[0])

# + colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1241318, "status": "ok", "timestamp": 1677008351228, "user": {"displayName": "amin norouzi", "userId": "02798455485104728835"}, "user_tz": 480} id="Z7HTdvp-jatT" outputId="e7223f19-b319-485b-d045-e09fb7082808"
# lst = [regular_df[regular_df.pointID == _].shape[0] for _ in regular_df.pointID.unique()]
# listt = pd.Series(lst)
# listt.value_counts()


# + colab={"base_uri": "https://localhost:8080/", "height": 232} executionInfo={"elapsed": 12, "status": "error", "timestamp": 1677022039324, "user": {"displayName": "amin norouzi", "userId": "02798455485104728835"}, "user_tz": 480} id="shmuk0yEmTl7" outputId="f81f5796-9093-40fa-bb8a-ec2eb55eb365"
# # keep the last n data for each pointID
# n = 105
# for _ in regular_df.pointID.unique():
#     regular_df[regular_df["pointID"] == _] =  regular_df[regular_df["pointID"] == _].tail(n)
#     # df = df.tail(29)

# + id="2G7-V6-znP7b"
# #  Pick a fields
# a_field = regular_df[regular_df.pointID==noJump_df.pointID.unique()[2]].copy()
# a_field.sort_values(by='human_system_start_time', axis=0, ascending=True, inplace=True)

# # Plot
# fig, ax = plt.subplots(1, 1, figsize=(20, 3), sharex='col', sharey='row',
#                        gridspec_kw={'hspace': 0.2, 'wspace': .05})
# ax.grid(True)
# ax.plot(a_field['human_system_start_time'],
#         a_field["indeks"],
#         linestyle='-', label="indeks", linewidth=3.5, color="dodgerblue", alpha=0.8)

# ax.legend(loc="lower right")
# plt.ylim([0, 1])


# + id="C_uODyrZnZYz"
# regular_df = regular_df[~regular_df.isnull()["indeks"]]

# + [markdown] id="j70PR2OwNLT4"
# #### Read imagery data from drive

# + id="FnncUBYMKtTI"



# + [markdown] id="A6_-JlL40Uwr"
# # Import Survey Data

# + id="uKszpc9Y1Fi7"
df = pd.read_csv('/content/drive/MyDrive/PhD/Bio_AgTillage/01. Codes/final_dataframe.csv', index_col="pointID")
df
