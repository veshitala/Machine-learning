"""
This uses only the train.csv to fit the data. Goal is to 
test incrementally whether additional features improve 
the roc_auc score
"""

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import gc

ROOT = 'main' + '_'

def build_model_input():

    print('[1] Read installment payments')
    inst = pd.read_csv('../input/installments_payments.csv')
    inst['payday_diff'] = inst.DAYS_INSTALMENT - inst.DAYS_ENTRY_PAYMENT

    print('Count payments')
    counts = inst.SK_ID_CURR.value_counts()
    counts = pd.DataFrame({'SK_ID_CURR':counts.index, 'NUM_PAYMENTS':counts.values})

    print('Count late payments')
    late_counts = inst[inst.payday_diff < 0].SK_ID_CURR.value_counts()
    late_counts = pd.DataFrame({'SK_ID_CURR': late_counts.index, 'NUM_LATE_PAYMENT':late_counts.values})

    print('Calculate fraction of late payments')
    agg_inst = counts.merge(late_counts, how='left', on='SK_ID_CURR')
    agg_inst['FRAC_LATE_PAY'] = agg_inst.NUM_LATE_PAYMENT/agg_inst.NUM_PAYMENTS
    agg_inst.fillna(0, inplace=True)

    agg_inst.columns = ['inst_' + f_ if f_ is not 'SK_ID_CURR' else f_ for f_ in agg_inst.columns]

    del inst, counts, late_counts
    gc.collect()

    print('[2] Read POS cash balance')
    pos = pd.read_csv('../input/POS_CASH_balance.csv')

    print('Convert categorical features')
    pos_dum = pd.get_dummies(pos.NAME_CONTRACT_STATUS, prefix='cs')
    pos_dum.drop(['cs_Active', 'cs_Completed'], axis=1, inplace=True)
    pos = pd.concat([pos.drop('NAME_CONTRACT_STATUS', axis=1), pos_dum], axis=1)

    pos.drop(['MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE'], axis=1, inplace=True)

    print('Compute average')
    nb_pos_per_curr = pos[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos['NUM_ENTRY'] = pos['SK_ID_CURR'].map(nb_pos_per_curr['SK_ID_PREV'])

    avg_pos = pos.groupby('SK_ID_CURR').mean()
    avg_pos.drop('SK_ID_PREV', axis=1, inplace=True)
    avg_pos.columns = ['pos_' + f_ for f_ in avg_pos.columns]

    del pos, nb_pos_per_curr
    gc.collect()

    print('[3] Read bureau data')
    buro = pd.read_csv('../input/bureau.csv')

    print('Convert categorical features')
    buro = pd.get_dummies(buro)
    del buro['CREDIT_CURRENCY_currency 1']

    print('Compute averages')
    avg_buro = buro.groupby('SK_ID_CURR').mean()
    del avg_buro['SK_ID_BUREAU']
    avg_buro.columns = ['buro_' + f_ for f_ in avg_buro.columns]

    print('[4] Read previous applications')
    prev = pd.read_csv('../input/previous_application.csv')
    prev.drop(['SK_ID_PREV', 
               'WEEKDAY_APPR_PROCESS_START', 
               'HOUR_APPR_PROCESS_START', 
               'NAME_CASH_LOAN_PURPOSE',
               'NAME_TYPE_SUITE',
               'NAME_CLIENT_TYPE',
              ], axis=1, inplace=True)

    print('Averaging approved loan statistics')
    prev_acc = prev[prev.NAME_CONTRACT_STATUS == 'Approved']\
                .drop(['FLAG_LAST_APPL_PER_CONTRACT',
                       'NAME_CONTRACT_STATUS',
                       'CODE_REJECT_REASON'], axis=1)
    avg_prev_acc = prev_acc.groupby('SK_ID_CURR').mean().reset_index()

    print('Averaging contract status')
    prev_cs = pd.DataFrame({'SK_ID_CURR':prev.SK_ID_CURR, 
                            'CONTRACT_STATUS': prev.NAME_CONTRACT_STATUS})
    prev_cs = pd.get_dummies(prev_cs, prefix='cs')
    avg_prev_cs = prev_cs.groupby('SK_ID_CURR').mean().reset_index()

    print('Merging statistics of approved loans and contract status')
    avg_prev = avg_prev_acc.merge(avg_prev_cs, how='left', on='SK_ID_CURR')
    avg_prev.columns = ['prev_' + f_ if f_ is not 'SK_ID_CURR' else f_ for f_ in avg_prev.columns]

    del prev, prev_acc, prev_cs, avg_prev_acc, avg_prev_cs
    gc.collect()


    print('Read training data')
    data = pd.read_csv('../input/application_train.csv')
    print('Shapes : ', data.shape)

    y = data['TARGET']
    del data['TARGET']

    data = pd.get_dummies(data)

    # Merge all data
    data = data.merge(right=agg_inst, how='left', on='SK_ID_CURR')
    data = data.merge(right=avg_prev, how='left', on='SK_ID_CURR')
    data = data.merge(right=avg_pos.reset_index(), how='left', on='SK_ID_CURR')
    data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

    del agg_inst, avg_pos, avg_buro
    gc.collect()

    return data, y

def train_model(data_, y_, folds_idx_):

    preds = np.zeros(data_.shape[0])

    feature_importance_df = pd.DataFrame()

    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]

    for n_fold, (trn_idx, val_idx) in enumerate(folds_idx_):
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        
        clf = LGBMClassifier(
            n_estimators=4000,
            learning_rate=0.03,
            num_leaves=30,
            colsample_bytree=.8,
            subsample=.9,
            max_depth=7,
            reg_alpha=.1,
            reg_lambda=.1,
            min_split_gain=.01,
            min_child_weight=2,
            silent=-1,
            verbose=-1,
        )
        
        clf.fit(trn_x, trn_y, 
                eval_set= [(trn_x, trn_y), (val_x, val_y)], 
                eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
               )

        preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, preds[val_idx])))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(y, preds))

    return preds, feature_importance_df

def display_importances(feature_importance_df_):

    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index
    
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    
    plt.figure(figsize=(8,10))
    sns.barplot(x="importance", y="feature", 
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('../output/' + ROOT+ 'lgbm_importances.png')


def display_roc_curve(y_, preds_, folds_idx_):

    plt.figure(figsize=(6,6))
    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):  
        fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], preds_[val_idx])
        score = roc_auc_score(y_.iloc[val_idx], preds_[val_idx])
        scores.append(score)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, preds_)
    score = roc_auc_score(y_, preds_)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig('../output/' + ROOT+ 'roc_curve.png')

def display_precision_recall(y_, preds_, folds_idx_):
    plt.figure(figsize=(6,6))

    scores = [] 
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        precision, recall, thresholds = precision_recall_curve(y_.iloc[val_idx], preds_[val_idx])
        score = average_precision_score(y_.iloc[val_idx], preds_[val_idx])
        scores.append(score)
        plt.plot(recall, precision, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))

    precision, recall, thresholds = precision_recall_curve(y_, preds_)
    score = average_precision_score(y_, preds_)
    plt.plot(recall, precision, color='b',
             label='Avg AP (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig('../output/' + ROOT+ 'recall_precision_curve.png')

if __name__ == '__main__':
    gc.enable()
    data, y = build_model_input()
    
    folds = KFold(n_splits=5, shuffle=True, random_state=42)
    folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds.split(data)]
    
    preds, importances = train_model(data, y, folds_idx)
    display_importances(feature_importance_df_=importances)
    display_roc_curve(y_=y, preds_=preds, folds_idx_=folds_idx)
    display_precision_recall(y_=y, preds_=preds, folds_idx_=folds_idx)
