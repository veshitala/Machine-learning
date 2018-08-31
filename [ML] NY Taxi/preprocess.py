import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('input/train.csv')
df = df[df.trip_duration < 7200]

print df.shape
exit()

# Check if the id's are unique => It is. Drop it.
print df.loc[:, 'id'].shape
print df.loc[:, 'id'].nunique()

# Convert "Datetime" string into datetime object
df.loc[:, 'pickup_datetime'] = pd.to_datetime(df.loc[:,'pickup_datetime'])
df.loc[:,'hour'] = df.pickup_datetime.dt.hour
df.loc[:,'day_of_week'] = df.pickup_datetime.dt.dayofweek

# Not necessary if you use xgboost or other tree-based algorithm
# df.loc[:,'hour'] = df.loc[:,'hour'].apply(lambda x: str(x))
# df.loc[:,'day_of_week'] = df.loc[:,'day_of_week'].apply(lambda x: str(x))

## Zoning

# Check the relative distribution of pickup and dropoff locations
# sample = df.sample(n=2500)

# plt.figure()
# plt.scatter(sample.pickup_longitude, sample.pickup_latitude,   s=3)
# plt.scatter(sample.dropoff_longitude, sample.dropoff_latitude, s=3)

# plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 50, n_init = 1, max_iter = 10, n_jobs=-1).\
    fit(df.loc[:, 'pickup_longitude':'pickup_latitude'])

df.loc[:,'pickup_zones'] = \
    kmeans.predict(df.loc[:, 'pickup_longitude':'pickup_latitude'])
df.loc[:,'dropoff_zones'] = \
    kmeans.predict(df.loc[:, 'dropoff_longitude':'dropoff_latitude'])

# sample = df.sample(n=2500)

# plt.figure()
# for k in xrange(50):
#     zone = sample[sample.pickup_zones == k]
#     plt.scatter(zone.pickup_longitude, zone.pickup_latitude,\
#                 s=3) 

# plt.figure()
# for k in xrange(50):
#     zone = sample[sample.dropoff_zones == k]
#     plt.scatter(zone.dropoff_longitude, zone.dropoff_latitude,\
#                 s=3) 

# plt.show()

# Calculate the cartesian distance between pick-up and drop-off locations
df.loc[:, 'distance'] = \
    np.sqrt((df.loc[:,'pickup_longitude'] - df.loc[:,'dropoff_longitude'])**2
           +(df.loc[:,'pickup_latitude']  - df.loc[:,'dropoff_latitude'])**2)

df.drop(['id','pickup_datetime','dropoff_datetime','store_and_fwd_flag'],\
    inplace=True, axis=1)

df.to_csv('input/processed_train.csv', index=False)

