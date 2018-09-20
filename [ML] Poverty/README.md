## Costa Rican Household Poverty Level Prediction
_tags: multiclass, tabular data_

Based on: https://www.kaggle.com/c/costa-rican-household-poverty-prediction

The input data files can be downloaded from the linked to the Kaggle website. 

- The goal is to predict the level of poverty based on features such as age, rent, education level, etc.

To model the data
- Individual members in households were aggregated (mean and stdev) to generate new features
- To correct for heavy imbalance between classes, Stratified KFold is used in the training step



