## Forest cover type

_tags: ensemble method, multiclass, tabular data_

Based on: https://www.kaggle.com/c/forest-cover-type-kernels-only

The input data files can be downloaded from the linked to the Kaggle website. 

- The goal is to predict the tree specie growing in a lot based on information about the lay of the land, soil type, hill shades and etc.

- We use PCA and manual methods to create new features

- We use lightGBM, extra random trees classifier, and random forest classfiers to generate scores that are used as features for the XGBoost classifier


