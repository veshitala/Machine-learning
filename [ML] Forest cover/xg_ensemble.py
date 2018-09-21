""" Forest cover type """

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import log_loss

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

ID = 'Id'
NFOLDS = 5
NUM_CLASS = 7
NROWS = None
SEED = 42

def generate_pca_features(df):

    pca   = PCA()
    X_pca = pca.fit_transform(StandardScaler().fit_transform(df.drop(ID, axis=1)))

    # Select number of features that can explain 95% of variance
    num_feature = (pca.explained_variance_ratio_.cumsum() < 0.90).argmin()

    df_pca = pd.DataFrame({ID: df[ID]})
    for i in range(num_feature):
        c_name = 'pca' + str(i)
        df_pca[c_name] = X_pca[:, i]

    return df_pca

def build_model_input():

    train = pd.read_csv('../input/train.csv', nrows=NROWS)
    test  = pd.read_csv('../input/test.csv', nrows=NROWS)

    all_data = pd.concat([train.drop('Cover_Type', axis=1), test], axis=0)

    print('Feature engineering')
    all_data['HF1'] = all_data['Horizontal_Distance_To_Hydrology'] + all_data['Horizontal_Distance_To_Fire_Points']
    all_data['HF2'] = all_data['Horizontal_Distance_To_Hydrology'] - all_data['Horizontal_Distance_To_Fire_Points']
    all_data['HR1'] = all_data['Horizontal_Distance_To_Hydrology'] + all_data['Horizontal_Distance_To_Roadways']
    all_data['HR2'] = all_data['Horizontal_Distance_To_Hydrology'] - all_data['Horizontal_Distance_To_Roadways']
    all_data['FR1'] = all_data['Horizontal_Distance_To_Fire_Points'] + all_data['Horizontal_Distance_To_Roadways']
    all_data['FR2'] = all_data['Horizontal_Distance_To_Fire_Points'] - all_data['Horizontal_Distance_To_Roadways']
    all_data['elev_vert']  = all_data['Elevation'] - all_data['Vertical_Distance_To_Hydrology']

    all_data['eucl_H'] = (all_data['Horizontal_Distance_To_Hydrology']**2 + all_data['Vertical_Distance_To_Hydrology']**2)**0.5
    all_data['mean_FHR'] = (all_data['Horizontal_Distance_To_Fire_Points']\
                          + all_data['Horizontal_Distance_To_Hydrology']\
                          + all_data['Horizontal_Distance_To_Roadways']) / 3

    print('Add PCA features')
    pca = generate_pca_features(all_data)
    all_data = all_data.merge(pca, on='Id')

    x_train = np.array(all_data.iloc[:len(train)])
    x_test = np.array(all_data.iloc[len(train):])

    y_train = np.array(train.Cover_Type - 1)
    print(y_train.shape)

    return x_train, x_test, y_train

class SKlearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def predict_proba(self, x):
        return self.clf.predict_proba(x)

class LgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.nrounds = params.pop('nrounds', 250)
        self.param = params
        self.param['seed'] = seed

    def train(self, x_train, y_train):
        dtrain = lgb.Dataset(x_train, label=y_train)
        self.gbdt = lgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(x)

    def predict_proba(self, x):
        return self.gbdt.predict(x, raw_score=True)

def get_oof(clf, x_train, x_test, y_train, kf):

    ntrain = len(x_train)
    ntest = len(x_test)

    oof_train = np.zeros((ntrain, NUM_CLASS))
    oof_test = np.zeros((ntest, NUM_CLASS))
    oof_test_skf = np.empty((NFOLDS, ntest, NUM_CLASS))

    for i, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
        print('Training fold {}'.format(i + 1))
        x_tr = x_train[train_idx]
        y_tr = y_train[train_idx]
        x_va = x_train[val_idx]

        clf.train(x_tr, y_tr)

        oof_train[val_idx] = clf.predict_proba(x_va)
        oof_test_skf[i, :] = clf.predict_proba(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train, oof_test

def train_1st_level(x_train, x_test, y_train, kf):

    et_params = {
        'n_jobs': -1,
        'n_estimators': 100,
        'criterion': 'entropy',
        'max_features': 0.5,
        'max_depth': 12,
        'min_samples_leaf': 2,
    }

    rf_params = {
        'n_jobs': -1,
        'n_estimators': 100,
        'criterion': 'entropy',
        'max_features': 0.2,
        'max_depth': 8,
        'min_samples_leaf': 2,
    }

    lgb_params = {
        'objective': 'softmax',
        'num_class': NUM_CLASS,
        'learning_rate': 0.075,
        'num_leaves': 30,
        'colsample_bytree': 0.7,
        'max_depth': 7,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_split_gain': 0.01,
        'min_child_weight': 1,
        'verbose': -1,
        'nrounds':350
    }

    print('Training 1st level')
    lg = LgbWrapper(seed=SEED, params=lgb_params)
    et = SKlearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
    rf = SKlearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)

    lg_oof_train, lg_oof_test = get_oof(lg, x_train, x_test, y_train, kf)
    et_oof_train, et_oof_test = get_oof(et, x_train, x_test, y_train, kf)
    rf_oof_train, rf_oof_test = get_oof(rf, x_train, x_test, y_train, kf)

    print("LG-CV: {}".format(accuracy_score(y_train, np.argmax(lg_oof_train, axis=1))))
    print("ET-CV: {}".format(accuracy_score(y_train, np.argmax(et_oof_train, axis=1))))
    print("RF-CV: {}".format(accuracy_score(y_train, np.argmax(rf_oof_train, axis=1))))

    x_train = np.concatenate([lg_oof_train, et_oof_train, rf_oof_train], axis=1)
    x_test = np.concatenate([lg_oof_test, et_oof_test, rf_oof_test], axis=1)

    print("{}, {}".format(x_train.shape, x_test.shape))

    return x_train, x_test

def train_2nd_level(x_train, x_test, y_train, kf):

    print('Training 2nd level')
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)

    xgb_params = {
        'seed': 0,
        'colsample_bytree': 0.8,
        'silent': 1,
        'subsample': 0.6,
        'learning_rate': 0.01,
        'objective': 'multi:softprob',
        'num_class': 7,
        'max_depth': 4,
        'num_parallel_tree': 1,
        'min_child_weight': 1,
        'eval_metric': 'mlogloss',
    }

    res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
                 early_stopping_rounds=25, verbose_eval=20, show_stdv=True)

    best_nrounds = res.shape[0] - 1
    cv_mean = res.iloc[-1, 0]
    cv_std  = res.iloc[-1, 1]

    print('Ensemble-CV: {0}+{1}'.format(cv_mean, cv_std))

    gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
    y_test = gbdt.predict(dtest).argmax(axis=1)

    return y_test

if __name__ == '__main__':
    
    target_names = ['Spruce/Fur', 
                    'Lodgepole pine', 
                    'Ponderosa Pine',
                    'Cottonwood/Willow',
                    'Aspen',
                    'Douglas-fir',
                    'Krummholz']

    # Build model inputs
    x_train, x_test, y_train = build_model_input()
    # Create Stratified Folds
    folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    #Train model
    x_train, x_test = train_1st_level(x_train, x_test, y_train, folds)
    y_test = train_2nd_level(x_train, x_test, y_train, folds)

    submission = pd.read_csv('../input/sample_submission.csv')
    submission.iloc[:, 1] = y_test
    submission.to_csv('submission.csv', index=None)    
