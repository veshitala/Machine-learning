## Forest cover type

_tags: ensemble/stacking method, multiclass, tabular data_

Based on: https://www.kaggle.com/c/forest-cover-type-kernels-only

The input data files can be downloaded from the linked to the Kaggle website. 

- The goal is to predict the tree specie growing in a lot based on information about the lay of the land, soil type, hill shades and etc.

- We use PCA and manual feature engeneering to generate new features

- We use lightGBM, ExtraTreeClassifier, and RandomForestClassifier to generate "scores" for each class and use those as input for xgboost to generate the predictions
