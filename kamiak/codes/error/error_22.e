/home/a.norouzikandelati/Projects/Tillage_mapping/codes/field_level_featureExtraction_eastWA_data.py:1068: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  Landsat_metricBased_df.insert(1, 'year', year_column)