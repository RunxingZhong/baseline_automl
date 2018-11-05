import pandas as pd
import numpy as np
import lightgbm as lgb
import hyperopt
from hyperopt import hp, tpe, STATUS_OK, space_eval, Trials
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from lib.util import timeit, log, Config
from typing import List, Dict
import datetime
import sys
import time


@timeit
def train(X: pd.DataFrame, y: pd.Series, config: Config):
    if "leak" in config:
        print('################## have leak ##################')
        return
    
    config['shape_5_0'] = X.shape[0]
    train_lightgbm(X, y, config)


@timeit
def predict(X: pd.DataFrame, config: Config) -> List:
    if "leak" in config:
        preds = predict_leak(X, config)
    else:
        preds = predict_lightgbm(X, config)

    return preds


@timeit
def validate(preds: pd.DataFrame, target_csv: str, mode: str) -> np.float64:
    df = pd.merge(preds, pd.read_csv(target_csv), on="line_id", left_index=True)
    score = roc_auc_score(df.target.values, df.prediction.values) if mode == "classification" else \
        np.sqrt(mean_squared_error(df.target.values, df.prediction.values))
    log("Score: {:0.4f}".format(score))
    return score


@timeit
def train_lightgbm(X: pd.DataFrame, y: pd.Series, config: Config):
    params = {
        "objective": "regression" if config["mode"] == "regression" else "binary",
        "metric": "rmse" if config["mode"] == "regression" else "auc",
        "verbosity": -1,
        "seed": 1,
    }

    X_sample, y_sample = data_sample(X, y)
    hyperparams = hyperopt_lightgbm(X_sample, y_sample, params, config)
   
    
    for i in range(1):
        print('################################################################## cv ' + str(i))
        t1_bagging = time.time()
        params['seed'] = i + 1
        # cv
        nfold = 5
        if config["mode"] == 'classification':
            skf = StratifiedKFold(n_splits = nfold, shuffle=True, random_state=777)
        else:
            skf = KFold(n_splits = nfold, shuffle=True, random_state=777)
        skf_split = skf.split(X, y)


        log('####################################################################### begin cv')
        log('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
        score_list = []
        config["model"] = []
        for fid, (train_idx, valid_idx) in enumerate(skf_split):
            t1_cv = time.time()
            print("FoldID:{}".format(fid))
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            dtrain = lgb.Dataset(X_train, label = y_train)
            dvalid = lgb.Dataset(X_valid, label = y_valid, reference = dtrain)

            cur_model = lgb.train({**params, **hyperparams}, dtrain, 3000, dvalid, early_stopping_rounds=50, verbose_eval=100)
            config["model"].append(cur_model)

            score_list.append(cur_model.best_score)
            # gc.collect()
            sys.stdout.flush()
            t2_cv = time.time()
            time_left = config.time_left()
            print('######### cv' + str(time_left))
            if (t2_cv - t1_cv) * (nfold - fid + 1) >= time_left:
                pass
                #break
            

        log('######################################################################### end cv')
        log('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))

        valid_auc = np.array([i['valid_0'][params['metric']] for i in score_list])
        print('valid', valid_auc, np.mean(valid_auc))
        cv_score = pd.DataFrame({'cv':np.hstack([valid_auc, np.mean(valid_auc)])})
        path = config['path_pred']
        print(path)
        cv_score.to_csv(path + '/cv_score_'+str(i)+'.csv', index = False)
        
        t2_bagging = time.time()
        time_left = config.time_left()
        print('#########bagging' + str(time_left))
        if (t2_bagging - t1_bagging) * 1.5 >= time_left:
            #break
            pass
    

@timeit
def predict_lightgbm(X: pd.DataFrame, config: Config) -> List:
    ans = []
    for model in config["model"]:
        p = model.predict(X, model.best_iteration)
        if config["non_negative_target"]:
            p = [max(0, i) for i in p]
        ans.append(p)
    
    if config["mode"] == "classification":
        print('average rank....')
        ans = pd.DataFrame(ans).T
        return ans.rank(method='dense').mean(axis = 1).values
    
    return np.mean(ans, axis = 0).flatten()


@timeit
def hyperopt_lightgbm(X: pd.DataFrame, y: pd.Series, params: Dict, config: Config):
    log('######################################################## begin hyperopt_lightgbm')
    log('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    
    X_train, X_val, y_train, y_val = data_split(X, y, test_size=0.5)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val)

    space = {
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.05),
        "max_depth": hp.choice("max_depth", [-1, 2, 3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", np.linspace(10, 200, 50, dtype=int)),
        "feature_fraction": hp.quniform("feature_fraction", 0.5, 1.0, 0.1),
        "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 1.0, 0.1),
        "bagging_freq": hp.choice("bagging_freq", np.linspace(0, 50, 10, dtype=int)),
        "reg_alpha": hp.uniform("reg_alpha", 0, 30),
        "reg_lambda": hp.uniform("reg_lambda", 0, 30),
        "min_child_weight": hp.uniform('min_child_weight', 0.5, 10),
    }
    
    def objective(hyperparams):
        model = lgb.train({**params, **hyperparams}, train_data, 300, valid_data,
                          early_stopping_rounds=100, verbose_eval=1000)

        score = model.best_score["valid_0"][params["metric"]]
        if config.is_classification():
            score = -score

        return {'loss': score, 'status': STATUS_OK}

    trials = Trials()
    best = hyperopt.fmin(fn=objective, space=space, trials=trials, algo=tpe.suggest, max_evals=50, verbose=0,
                         rstate=np.random.RandomState(777))

    hyperparams = space_eval(space, best)
    log('########################################################## end hyperopt_lightgbm')
    log('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    log("{:0.4f} {}".format(trials.best_trial['result']['loss'], hyperparams))
    return hyperparams


@timeit
def predict_leak(X: pd.DataFrame, config: Config) -> List:
    preds = pd.Series(0, index=X.index)

    for name, group in X.groupby(by=config["leak"]["id_col"]):
        gr = group.sort_values(config["leak"]["dt_col"])
        preds.loc[gr.index] = gr[config["leak"]["num_col"]].shift(config["leak"]["lag"])

    return preds.fillna(0).tolist()


def data_split(X: pd.DataFrame, y: pd.Series, test_size: float=0.2) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    return train_test_split(X, y, test_size=test_size, random_state=1)


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000) -> (pd.DataFrame, pd.Series):
    if len(X) > nrows:
        X_sample = X.sample(nrows, random_state=1)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    return X_sample, y_sample
