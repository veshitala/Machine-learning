import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('input/processed_test.csv')

## Convert zones into categorical variable
df.loc[:,'pickup_zones']  = df.loc[:,'pickup_zones'].apply(lambda x: str(x))
df.loc[:,'dropoff_zones'] = df.loc[:,'dropoff_zones'].apply(lambda x: str(x))
df.loc[:,'day_of_week'] = df.loc[:,'day_of_week'].apply(lambda x: str(x))

# Convert pickup and dropoff zones into dummy variables
df = pd.get_dummies(df)

df_short = df.sample(n=1000)
y_train = df_short.trip_duration.values
X_train = df_short.drop(['trip_duration'], axis=1)
X_train = X_train.values

## Xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer

def rmsle(actual, predicted):
    assert(len(predicted) == len(actual))
    # Force negative prediction to be 0
    predicted[predicted < 0] = 0
    p = np.log(predicted + 1)
    a = np.log(actual + 1)
    return (((p - a)**2).sum() / len(predicted))**0.5

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

print "\nTraining xgboost.."

xgb_model = xgb.XGBRegressor()

alphas = np.around(np.logspace(-1,1, 10), decimals=3)
depths = np.arange(2, 10, 1)
etas = [0.2,0.3,0.4]

start = time.time()

clf = GridSearchCV(xgb_model, 
                   {'reg_alpha': alphas,
                    'max_depth': depths,
                    'learning_rate': etas
                    }, verbose=1,
                    scoring = rmsle_scorer )

clf.fit(X_train, y_train)

end = time.time()
print(clf.best_score_)
print(clf.best_params_)
print 'elapsed time: {0:.1f} s'.format(end - start)

# ## Contour plot of gridsearch result
# scores = clf.cv_results_['mean_test_score'].\
#     reshape(len(depths), len(alphas))

# plt.figure(figsize=(8,6))
# plt.imshow(scores, cmap='hot')
# # First variable
# plt.xlabel('reg_alpha')
# # Second variable
# plt.ylabel('max_depth')
# plt.colorbar()
# plt.xticks(np.arange(len(alphas)), alphas, rotation=45)
# plt.yticks(np.arange(len(depths)), depths)
# plt.title('Validation accuracy')
# # plt.show()


best_alpha = clf.best_params_['reg_alpha']
best_depth = clf.best_params_['max_depth']
best_eta   = clf.best_params_['learning_rate']

## Use the best params to fit the whole data set

start = time.time()

y_train = df.trip_duration.values
X_train = df.drop(['trip_duration'], axis=1)
X_train = X_train.values

# Manually split training example into training and validation group
Xtr, Xv, ytr, yv = \
    train_test_split(X_train, y_train, test_size=0.2)

params = {}
params['reg_alpha']     = best_alpha
params['max_depth']     = best_depth
params['learning_rate'] = best_eta

model = xgb.XGBRegressor()
model.fit(Xtr, ytr)

ypred = model.predict(Xv)
ypred[ypred < 0] = 0

end = time.time()

print "rmsle", rmsle(ypred, yv)
print 'elapsed time: {0:.1f} min'.format((end - start)/60.0)

# plt.show()