## Home Credit Default Risk

_tags: classification, large dataset, relational database_

Based on: https://www.kaggle.com/c/home-credit-default-risk

The input data files can be downloaded from the linked to the Kaggle website. 

- The goal is to predict the likelihood of loan candidates defaulting based on their credit and loan payment history

- Customer data and credit histories are distributed over multiple databases that were indexed like an SQL database that had to be aggregated to generate features.

- A gradient boosted tree method (LightGBM) is used to model the data

- Code is based on the kernel posted at https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm
