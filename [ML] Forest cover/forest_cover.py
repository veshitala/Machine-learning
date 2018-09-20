""" Forest cover type """

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from util import *

NUM_CLASS = 7

def build_model_input():

    train = pd.read_csv('../input/train.csv')
    test  = pd.read_csv('../input/test.csv')

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

    data = all_data.iloc[:len(train)]
    test = all_data.iloc[len(train):]

    y = train.Cover_Type - 1

    return data, test, y

def train_model(data_, test_, y_, folds_):

    print(y_.shape)


    data_pred = np.zeros(data_.shape[0])
    test_prob = np.zeros((test_.shape[0], NUM_CLASS))

    feature_importance_df = pd.DataFrame()

    feats = [f for f in data_.columns if f not in ['Id']]

    print('Training model')
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
    
        clf = LGBMClassifier(n_estimators=4000,
                             learning_rate=0.05,
                             num_leaves=30,
                             colsample_bytree=1.0,
                             max_depth=7,
                             reg_alpha=0.1,
                             reg_lambda=0.1,
                             min_split_gain=0.01,
                             min_child_weight=2,
                             silent=-1,
                             verbose=-1,
                             objective='multiclassova',
                             num_class=NUM_CLASS)

        clf.fit(trn_x, trn_y,
                eval_set=[(trn_x, trn_y), (val_x, val_y)],
                verbose=100, early_stopping_rounds=100
                )

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


        data_pred[val_idx] = clf.predict(val_x, num_iteration=clf.best_iteration_)
        test_prob += clf.predict_proba(test_[feats], num_iteration=clf.best_iteration_) / folds_.n_splits

        print('Fold %2d macro f1_score : ' % (n_fold + 1), f1_score(val_y, data_pred[val_idx], average=None))
        print('Fold %2d accuracy score : ' % (n_fold + 1), accuracy_score(val_y, data_pred[val_idx]))

    print('Full macro f1_score:', f1_score(y_, data_pred, average=None))
    print('Full accuracy:', accuracy_score(y_, data_pred))

    test_['Cover_Type'] = np.argmax(test_prob, axis=1) + 1   # categories are labeled from 1-4

    print(clf)
    
    print(len(feats), feature_importance_df.shape)

    return data_pred, test_[['Id', 'Cover_Type']], feature_importance_df

def display_importance(feature_importance_df_):
    
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:30].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(10,6))
    sns.barplot(x='importance', y='feature',
                data=best_features.sort_values(by="importance", ascending=False))
    plt.savefig('feature_importance.png')    

def display_confusion_matrix(y_pred_, y_true_, target_names):
    cm = confusion_matrix(y_pred=y_pred_, y_true=y_true_)
    plot_confusion_matrix(cm, normalize=True, target_names=target_names)

def display_label_distribution(train_pred, test_pred):

    nbins = 7
    labels = ['Train', 'Test']
    
    plt.figure(figsize=(6,6))
    plt.hist([train_pred, test_pred], nbins, normed=True, label=labels)
    plt.legend(prop={'size': 10})
    plt.title('label distribution')
    plt.savefig('label_distribution.png')

if __name__ == '__main__':
    
    target_names = ['Spruce/Fur', 
                    'Lodgepole pine', 
                    'Ponderosa Pine',
                    'Cottonwood/Willow',
                    'Aspen',
                    'Douglas-fir',
                    'Krummholz']

    # Build model inputs
    data, test, y = build_model_input()
    # Create Stratified Folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #Train model
    data_pred, test_pred, importances = train_model(data, test, y, folds)

    # Save test predictions
    test_pred.to_csv('sample_submission.csv', index=False)
    # Display few graphs
    display_importance(feature_importance_df_=importances)
    display_confusion_matrix(data_pred, y, target_names)
    display_label_distribution(data_pred + 1, test_pred.Cover_Type.values)

    # Evaluate macro f1_score for training data
    print('Macro f1-score is.. ', f1_score(y, data_pred, average=None))
    print('Accuracy is.. ', accuracy_score(y, data_pred))


    plt.show()

