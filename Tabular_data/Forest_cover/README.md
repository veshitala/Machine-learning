## Forest cover type

_tags: ensemble method, multiclass, tabular data_

Based on: https://www.kaggle.com/c/forest-cover-type-kernels-only

The input data files can be downloaded from the linked to the Kaggle website. 

- The goal is to predict the tree specie growing in a lot based on information about the lay of the land, soil type, hill shades and etc.

- We use PCA and manual methods to create new features

- We use _stacking_ to combine the output (class probability) generated via lightGBM, extra random trees, and random forest classfiers as features for the XGBoost classifier

- Each model (lightGBM, ExtraTrees, RandomForest) are sensitive to slightly different elements of the input data. By combining the results from different models, we are able to get a model that has better predictive power. 


