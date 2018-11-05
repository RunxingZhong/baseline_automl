import copy
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lib.features import select_features
from lib.util import timeit, log, Config


@timeit
def preprocess(df: pd.DataFrame, config: Config):
    # featuer selecrtion
    feature_selection(df, config)
    
    # transform
    preprocess_pipeline(df, config)
    

def preprocess_pipeline(df: pd.DataFrame, config: Config):
    if leak_detect(df, config):
        return

    ##
    log(df.shape)
  
#     get_nan_fea(df, config)
#     log(df.shape)
    
    drop_isnull(df, config)
    log(df.shape)
    
    drop_columns(df, config)
    log(df.shape)
    
    fillna(df, config)
    log(df.shape)
    
    #add_delta(df, config)
    
    to_int8(df, config)
    log(df.shape)
    
    non_negative_target_detect(df, config)
    log(df.shape)
    
    subsample(df, config, max_size_mb=2 * 1024)
    log(df.shape)

                              
    ##
    #get_time_diff(df, config)
    #log(df.shape)
    
    transform_datetime(df, config)
    log(df.shape)
    
    
#     transform_num_cat(df, config)
#     log(df.shape)
    
#     transform_categorical_2(df, config)
#     log(df.shape)
    
    transform_categorical(df, config)
    log(df.shape)
    
    scale(df, config)
    log(df.shape)
    
    subsample(df, config, max_size_mb=2 * 1024)
    log(df.shape)

    
@timeit
def add_delta(df: pd.DataFrame, config: Config):
    if "delta" not in config:
        config["delta"] = []
        data_df = [c for c in df if c.startswith("datetime_")]

        if len(data_df) == 0:
            return
        if len(data_df) == 1:
            print("--------do add delta----------")
            df.sort_values(data_df[0])
            for c in [c for c in df if c.startswith("number_")]:
                if df[c].nunique() > 100:
                    tmp_delta = c+"_delta"
                    df[tmp_delta] = df[c]
                    df[tmp_delta] = df[tmp_delta].shift(1)
                    df[tmp_delta].fillna(0,inplace=True)
                    df[tmp_delta] -= df[c]
                    config["delta"].append(c)
    else:
        if len(config["delta"])>0:
            for c in config["delta"]:
                tmp_delta = c+"_delta"
                df[tmp_delta] = df[c]
                df[tmp_delta] = df[tmp_delta].shift(1)
                df[tmp_delta].fillna(0,inplace=True)
                df[tmp_delta] -= df[c]    
    
@timeit
def get_nan_fea(df, config):
    col_num = [c for c in df if c.startswith('number_')]
    col_str = [c for c in df if c.startswith('string_')]
    col_time = [c for c in df if c.startswith('datetime_')]
    col_id = [c for c in df if c.startswith('id_')]

    df2 = pd.DataFrame([])
    if 'nan_fea' not in config:
        config['nan_fea'] = {}
        df2['nan_sum_num'] = df[col_num].isnull().sum(axis = 1)
        if (df2['nan_sum_num'] > 0).sum() == 0:
            df2.drop(['nan_sum_num'], axis = 1, inplace = True)
        if 'nan_sum_num' in df2:
            config['nan_fea']['nan_sum_num'] = 1

        df2['nan_sum_str'] = df[col_str].isnull().sum(axis = 1)
        if (df2['nan_sum_str'] > 0).sum() == 0:
            df2.drop(['nan_sum_str'], axis = 1, inplace = True)
        if 'nan_sum_str' in df2:
            config['nan_fea']['nan_sum_str'] = 1

        df2['nan_sum_time'] = df[col_time].isnull().sum(axis = 1)
        if (df2['nan_sum_time'] > 0).sum() == 0:
            df2.drop(['nan_sum_time'], axis = 1, inplace = True)
        if 'nan_sum_time' in df2:
            config['nan_fea']['nan_sum_time'] = 1

        df2['nan_sum_id'] = df[col_id].isnull().sum(axis = 1)
        if (df2['nan_sum_id'] > 0).sum() == 0:
            df2.drop(['nan_sum_id'], axis = 1, inplace = True)
        if 'nan_sum_id' in df2:
            config['nan_fea']['nan_sum_id'] = 1
    else:
        if 'nan_sum_num' in config['nan_fea']:
            df2['nan_sum_num'] = df[col_num].isnull().sum(axis = 1)
        if 'nan_sum_str' in config['nan_fea']:
            df2['nan_sum_str'] = df[col_str].isnull().sum(axis = 1)
        if 'nan_sum_time' in config['nan_fea']:
            df2['nan_sum_time'] = df[col_time].isnull().sum(axis = 1)
        if 'nan_sum_id' in config['nan_fea']:
            df2['nan_sum_id'] = df[col_id].isnull().sum(axis = 1)

    print(len(config['nan_fea']))
    if len(config['nan_fea']) > 1:
        df2['nan_sum_all'] = df2.sum(axis = 1)

    print(len(config['nan_fea']))    
    if len(config['nan_fea']) > 0:
        df2['nan_ratio'] = df2.sum(axis = 1) / df.shape[1]
        df[df2.columns] = df2[df2.columns]
    
 
@timeit
def drop_isnull(df: pd.DataFrame, config: Config):
    if 'drop_isnull' not in config:
        log('find it')
        col_drop = df.columns[df.isnull().sum() / df.shape[0] > 0.99]
        config['drop_isnull'] = col_drop
    
    log(len(config["drop_isnull"]))
    if len(config['drop_isnull']) > 0:
        log('do it')
        df.drop(config['drop_isnull'], axis = 1, inplace = True)
    
@timeit
def drop_columns(df: pd.DataFrame, config: Config):
    df.drop([c for c in ["is_test", "line_id"] if c in df], axis=1, inplace=True)
    log('drop line_id')
    drop_constant_columns(df, config)


    
@timeit
def drop_constant_columns(df: pd.DataFrame, config: Config):
    if "constant_columns" not in config:
        log('find it')
        config["constant_columns"] = [c for c in df if c.startswith("number_") and not (df[c] != df[c].iloc[0]).any()]
        log("Constant columns: " + ", ".join(config["constant_columns"]))

    log(len(config["constant_columns"]))
    if len(config["constant_columns"]) > 0:
        log('do it')
        df.drop(config["constant_columns"], axis=1, inplace=True)    
   

@timeit
def fillna(df: pd.DataFrame, config: Config):
    cnt1, cnt2, cnt3, cnt4 = 0, 0, 0, 0
    
    for c in [c for c in df]:
        if c.startswith("number_"):
            cnt1 += 1
            df[c].fillna(-7777777, inplace=True)
        
        if c.startswith("string_"):
            cnt2 += 1
            df[c].fillna("7", inplace=True)
        
        if c.startswith("datetime_"):
            cnt3 += 1
            df[c].fillna(datetime.datetime(1970, 1, 1), inplace=True)
        
        if c.startswith("id"):
            cnt4 += 1
            df[c].fillna(7777777, inplace=True)
    
    log("nan: number, string, datetime, id = {}, {}, {}, {}".format(cnt1, cnt2, cnt3, cnt4))
  

@timeit
def to_int8(df: pd.DataFrame, config: Config):
    if "int8_columns" not in config:
        log('find it')
        config["int8_columns"] = []
        vals = [-1, 0, 1]

        for c in [c for c in df if c.startswith("number_")]:
            if (~df[c].isin(vals)).any():
                continue
            config["int8_columns"].append(c)
        
        
        log(config["int8_columns"])
    
    log(len(config["int8_columns"]))
    if len(config["int8_columns"]) > 0:
        log('do it')
        df.loc[:, config["int8_columns"]] = df.loc[:, config["int8_columns"]].astype(np.int8)        
        

        
@timeit
def non_negative_target_detect(df: pd.DataFrame, config: Config):
    if config.is_train():
        config["non_negative_target"] = df["target"].lt(0).sum() == 0
   

@timeit
def get_time_diff(df: pd.DataFrame, config: Config):
    if 'time_diff' not in config:
        print('find it')
        config['time_diff'] = []
        col_time = [c for c in df if c.startswith("datetime_")]
        for i, c1 in enumerate(col_time):
            if i >= 5:
                break
            for j, c2 in enumerate(col_time):
                if j < i:
                    continue
                if j >= 10:
                    break    
                config['time_diff'].append('time-diff-' + c1 + '-' + c2)
   
    log(len(config['time_diff']))
    log(config['time_diff'])
    if len(config['time_diff']) > 0:
        print('do it')
        for col in config['time_diff']:
            cols = col.split('-')
            df[col] = (df[cols[2]] - df[cols[3]]).apply(lambda x: x.days)
            df[col + '_month'] = df[col].apply(lambda x: int(x / 30))
            df[col + '_week'] = df[col].apply(lambda x: int(x / 7))

import time
@timeit
def transform_datetime(df: pd.DataFrame, config: Config):
    date_parts = ["year", "weekday", "month", "day", "hour"]

    if "date_columns" not in config:
        log('find it')
        config["date_columns"] = {}

        for c in [c for c in df if c.startswith("datetime_")]:   
#             if config['df_size_mb'] > 1024:
#                 df.drop(c, axis=1, inplace=True) # drop origin datetime
#                 continue
                
                
            config["date_columns"][c] = []
            for part in date_parts:
                part_col = c + "_" + part # datetime_0_yrer
                
                if config['df_size_mb'] > 1024:
                    df[part_col] = getattr(pd.to_datetime(df[c], format = '%Y-%m-%d').dt, part).astype(np.uint16 if part == "year" else np.uint8).values
                    
                else:
                    df[part_col] = getattr(df[c].dt, part).astype(np.uint16 if part == "year" else np.uint8).values
                
                if not (df[part_col] != df[part_col].iloc[0]).any(): # constant value
                    log(part_col + " is constant")
                    df.drop(part_col, axis=1, inplace=True)
                else:
#                     if part == 'weekday':
#                         df[part_col + '_is67'] = df[part_col].apply(lambda x: 1 if x == 6 or x == 7 else 0)
                    config["date_columns"][c].append(part)

            df.drop(c, axis=1, inplace=True) # drop origin datetime
    else:
        log('do it')
        for c, parts in config["date_columns"].items():
            for part in parts:
                part_col = c + "_" + part
                if config['df_size_mb'] > 1024:
                     df[part_col] = getattr(pd.to_datetime(df[c], format = '%Y-%m-%d').dt, part).astype(np.uint16 if part == "year" else np.uint8).values
                else:
                    df[part_col] = getattr(df[c].dt, part)
#             if part == 'weekday':
#                 df[part_col + '_is67'] = df[part_col].apply(lambda x: 1 if x == 6 or x == 7 else 0)
            df.drop(c, axis=1, inplace=True)

    
    cnt_old = 0
    cnt_new = 0
    for c, parts in config["date_columns"].items():
        cnt_old += 1
        for part in parts:
            cnt_new += 1
    log("col, new = {}, {}".format(cnt_old, cnt_new))

@timeit    
def transform_num_cat(df: pd.DataFrame, config: Config):
    if "num_cat_col" not in config:
        # https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
        prior = config["num_cat_col_prior"] = df["target"].mean()
        min_samples_leaf = int(0.01 * len(df))
        smoothing = 0.5 * min_samples_leaf

        config["num_cat_col"] = {}
        col_nunique = df.nunique()
        col_num_cat = df.columns[col_nunique < 50]
        for c in col_num_cat:
            if not c.startswith("number"):
                continue
            averages = df[[c, "target"]].groupby(c)["target"].agg(["mean", "count"])
            smooth = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
            averages["target"] = prior * (1 - smooth) + averages["mean"] * smooth
            
            averages["target"] = averages["count"]
            config["num_cat_col"][c] = averages["target"].to_dict()

        log(list(config["num_cat_col"].keys()))

    for c, values in config["num_cat_col"].items():
        df.loc[:, c + '_target_encoding'] = df[c].apply(lambda x: values[x] if x in values else config["num_cat_col_prior"])    
        
@timeit
def transform_categorical_2(df: pd.DataFrame, config: Config):
    if "categorical_columns_2" not in config:
        config["categorical_columns_2"] = {}
        for c in [c for c in df if c.startswith("string_")]:
            k = np.unique(df[c])
            v = range(len(k))
            config["categorical_columns_2"][c] = dict(zip(k, v))

        log(list(config["categorical_columns_2"].keys()))

    for c, values in config["categorical_columns_2"].items():
        df.loc[:, 'label_encoding_' + c] = df[c].apply(lambda x: values[x] if x in values else -1)

@timeit
def transform_categorical(df: pd.DataFrame, config: Config):
    if "categorical_columns" not in config:
        # https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
        prior = config["categorical_prior"] = df["target"].mean()
        min_samples_leaf = int(0.01 * len(df))
        smoothing = 0.5 * min_samples_leaf

        config["categorical_columns"] = {}
        for c in [c for c in df if c.startswith("string_")]:
            averages = df[[c, "target"]].groupby(c)["target"].agg(["mean", "count"])
            smooth = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
            averages["target"] = prior * (1 - smooth) + averages["mean"] * smooth
            config["categorical_columns"][c] = averages["target"].to_dict()

        log(list(config["categorical_columns"].keys()))

    for c, values in config["categorical_columns"].items():
        df.loc[:, c] = df[c].apply(lambda x: values[x] if x in values else config["categorical_prior"])        

@timeit
def scale(df: pd.DataFrame, config: Config):
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    scale_columns = [c for c in df if c.startswith("number_") and df[c].dtype != np.int8 and
                     c not in config["categorical_columns"]]

    if len(scale_columns) > 0:
        if "scaler" not in config:
            config["scaler"] = StandardScaler(copy=False)
            config["scaler"].fit(df[scale_columns])

        df[scale_columns] = config["scaler"].transform(df[scale_columns])



@timeit
def subsample(df: pd.DataFrame, config: Config, max_size_mb: float=1.0):
    if config.is_train():
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if df_size_mb > max_size_mb:
            mem_per_row = df_size_mb / len(df)
            sample_rows = int(max_size_mb / mem_per_row)

            log("Size limit exceeded: {:0.2f} Mb. Dataset rows: {}. Subsample to {} rows.".format(df_size_mb, len(df), sample_rows))
            _, df_drop = train_test_split(df, train_size=sample_rows, random_state=1)
            df.drop(df_drop.index, inplace=True)

            config["nrows"] = sample_rows
        else:
            config["nrows"] = len(df)



@timeit
def feature_selection(df: pd.DataFrame, config: Config):
    if config.is_train():
        df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if df_size_mb < 1 * 1024:
            log('smaller than 2 GB')
            return
        
        log('bigger than 2 GB')
        
        selected_columns = []
        config_sample = copy.deepcopy(config)
        for i in range(10):
            df_sample = df.sample(min(1000, len(df)), random_state=i).copy(deep=True)
            preprocess_pipeline(df_sample, config_sample)
            y = df_sample["target"]
            X = df_sample.drop("target", axis=1)

            if len(selected_columns) > 0:
                X = X.drop(selected_columns, axis=1)

            if len(X.columns) > 0:
                selected_columns += select_features(X, y, config["mode"])
            else:
                break

        log("Selected columns: {}".format(selected_columns))

        drop_number_columns = [c for c in df if (c.startswith("number_") or c.startswith("id_")) and c not in selected_columns]
        if len(drop_number_columns) > 0:
            config["drop_number_columns"] = drop_number_columns

        config["date_columns"] = {}
        for c in [c for c in selected_columns if c.startswith("datetime_")]:
            d = c.split("_")
            date_col = d[0] + "_" + d[1]
            date_part = d[2]

            if date_col not in config["date_columns"]:
                config["date_columns"][date_col] = []

            config["date_columns"][date_col].append(date_part)

        drop_datetime_columns = [c for c in df if c.startswith("datetime_") and c not in config["date_columns"]]
        if len(drop_datetime_columns) > 0:
            config["drop_datetime_columns"] = drop_datetime_columns

    if "drop_number_columns" in config:
        log("Drop number columns: {}".format(config["drop_number_columns"]))
        df.drop(config["drop_number_columns"], axis=1, inplace=True)

    if "drop_datetime_columns" in config:
        log("Drop datetime columns: {}".format(config["drop_datetime_columns"]))
        df.drop(config["drop_datetime_columns"], axis=1, inplace=True)


@timeit
# https://github.com/bagxi/sdsj2018_lightgbm_baseline
# https://forum-sdsj.datasouls.com/t/topic/304/3
def leak_detect(df: pd.DataFrame, config: Config) -> bool:
    if config.is_predict():
        return "leak" in config

    id_cols = [c for c in df if c.startswith('id_')]
    dt_cols = [c for c in df if c.startswith('datetime_')]

    if id_cols and dt_cols:
        num_cols = [c for c in df if c.startswith('number_')]
        for id_col in id_cols:
            group = df.groupby(by=id_col).get_group(df[id_col].iloc[0])

            for dt_col in dt_cols:
                sorted_group = group.sort_values(dt_col)

                for lag in range(-1, -10, -1):
                    for col in num_cols:
                        corr = sorted_group['target'].corr(sorted_group[col].shift(lag))
                        if corr >= 0.99:
                            config["leak"] = {
                                "num_col": col,
                                "lag": lag,
                                "id_col": id_col,
                                "dt_col": dt_col,
                            }
                            return True

    return False
