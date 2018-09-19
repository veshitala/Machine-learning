## Costa Rican Household Poverty Level Prediction
_tags: multiclass_

Based on: https://www.kaggle.com/c/costa-rican-household-poverty-prediction

The input data files can be downloaded from the linked to the Kaggle website. 

- The goal is to predict the level of poverty based on features such as age, rent, education level, etc.

To model the data
- individual members in households were aggregated (mean and stdev) to generate new features
- Stratified KFold is used to generate class weight due to heavy imbalance between classes.



