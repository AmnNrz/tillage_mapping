path_to_data <-  paste0('/Users/aminnorouzi/Library/CloudStorage/',
                'OneDrive-WashingtonStateUniversity(email.wsu.edu)/',
                'Ph.D/Projects/Tillage_Mapping/Data/')

path_to_cdl_data <-  paste0(path_to_data,
  "MAPPING_DATA_2011_2012_2022/2012_2017_2022/cdl_data/"
)

path_to_landsat_data = paste0(path_to_data,
  "MAPPING_DATA_2011_2012_2022/2012_2017_2022/landsat_data/"
)

path_to_concatenated_data = paste0(path_to_data,
  "MAPPING_DATA_2011_2012_2022/2012_2017_2022/"
)

lsat_12_17_22 = read.csv(paste0(
  path_to_concatenated_data, "concatenated_landsat_file.csv"))
cdl_12_17_22 = read.csv(paste0(
  path_to_concatenated_data, "concatenated_cdl_file.csv"))
