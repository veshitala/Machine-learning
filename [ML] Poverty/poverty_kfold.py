import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from util import plot_confusion_matrix


def fix_target(train):
    """
    The poverty level of members of some households are inconsistent.
    This fixes that.
    """
    household_std = train.groupby('idhogar').std()
    mislabeled = household_std[household_std.Target > 0]
    
    # parentesco1 == 1: he/she is the head of household
    y = train[['Id', 'idhogar', 'parentesco1','Target']]

    for household in mislabeled:
        members = y[y.idhogar == household]
        head = members[members.parentesco1 == 1]
        if len(head) == 0:
            continue
        else:
            y.loc[members.index, 'Target'] = head.Target.iloc[0]

    return y.Target

def build_model_input():

    train = pd.read_csv('../input/train.csv')
    test  = pd.read_csv('../input/test.csv')

    all_data = pd.concat([train.drop('Target', axis=1), test], axis=0)

    print('Drop squared features')
    all_data = all_data.drop(['SQBescolari', 
                              'SQBage', 
                              'SQBhogar_total',
                              'SQBedjefe',
                              'SQBhogar_nin',
                              'SQBovercrowding',
                              'SQBdependency',
                              'SQBmeaned',
                              'agesq'], axis=1)

    print('Replace missing values')
    all_data.v2a1.fillna(-1, inplace=True)         # monthly rent payment
    all_data.v18q1.fillna(0, inplace=True)         # num of tablets
    all_data.rez_esc.fillna(-1, inplace=True)      # years behind school

    print('Replace "yes" and "no" in entries')
    all_data.replace(to_replace='yes', value=1, inplace=True)
    all_data.replace(to_replace='no', value=0, inplace=True)

    print('Correct dtypes for "dependency", "edjefe/a"')
    all_data.dependency = all_data.dependency.astype('float')
    all_data.edjefe     = all_data.edjefe.astype('int64')
    all_data.edjefa     = all_data.edjefa.astype('int64')

    print('Aggregate the variables (mean and stdev)')
    household_mean = all_data.groupby('idhogar').mean()
    household_std  = all_data.groupby('idhogar').std()

    data = pd.DataFrame({'Id':train.Id, 'idhogar': train.idhogar})
    test = pd.DataFrame({'Id':test.Id, 'idhogar': test.idhogar})

    columns = household_mean.columns

    print('Generate new features for the data frames')
    for f_ in columns:
        data['mean_' + f_] = data['idhogar'].map(household_mean[f_])
        data['std_' + f_]  = data['idhogar'].map(household_std[f_])
        test['mean_' + f_] = test['idhogar'].map(household_mean[f_])
        test['std_' + f_]  = test['idhogar'].map(household_std[f_])
        
    y = fix_target(train) - 1
    
    return data, test, y

def train_model(data_, test_, y_, folds_):

    print(y_.shape)


    data_pred = np.zeros(data_.shape[0])
    test_prob = np.zeros((test_.shape[0], 4))

    feature_importance_df = pd.DataFrame()

    feats = [f for f in data_.columns if f not in ['Id', 'idhogar', 'Target']]

    print('Training model')
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_, y_)):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
    
        clf = LGBMClassifier(n_estimators=4000,
                             learning_rate=0.03,
                             num_leaves=30,
                             colsample_bytree=0.8,
                             max_depth=7,
                             # reg_alpha=0.1,
                             # reg_lambda=0.1,
                             min_split_gain=0.0,
                             # min_child_weight=2,
                             silent=-1,
                             verbose=-1,
                             objective='multiclassova',
                             num_class=4,
                             class_weight='balanced')

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

    print('Full macro f1_score:', f1_score(y_, data_pred, average=None))

    test_['Target'] = np.argmax(test_prob, axis=1) + 1   # categories are labeled from 1-4

    print(clf)

    
    print(len(feats), feature_importance_df.shape)

    return data_pred, test_[['Id', 'Target']], feature_importance_df

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

    nbins = 4
    labels = ['Train', 'Test']
    
    plt.figure(figsize=(6,6))
    plt.hist([train_pred, test_pred], nbins, normed=True, label=labels)
    plt.legend(prop={'size': 10})
    plt.title('label distribution')
    plt.savefig('label_distribution.png')

if __name__ == '__main__':
    
    target_names = ['extreme pov.',
                    'moderate pov.',
                    'vulnerable',
                    'non-vulnerable']    

    # Build model inputs
    data, test, y = build_model_input()
    # Create Folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Train model
    data_pred, test_pred, importances = train_model(data, test, y, folds)
    # Save test predictions
    test_pred.to_csv('sample_submission.csv', index=False)
    # Display few graphs
    display_importance(feature_importance_df_=importances)
    display_confusion_matrix(data_pred, y, target_names)
    display_label_distribution(data_pred + 1, test_pred.Target.values)

    # Evaluate macro f1_score for training data
    score = f1_score(y, data_pred, average=None)
    print('Macro f1-score is.. ', score)

    plt.show()
