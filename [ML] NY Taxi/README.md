## NY Taxi duration

_tags: regression, feature engineering

Based on: https://www.kaggle.com/c/nyc-taxi-trip-duration

The input data files can be downloaded from the linked to the Kaggle website. 

- The goal is to predict the NY trip duration based on the trip date and pick-up/drop off location

- We generate additional features such as pick-up/drop-off zones and day-of-week/time-of-day features.

- Finally, XGBoost is used to carry out regression on the input data.

- *preprocess.py* generates additional features / *main.py* fits the xgboost model
