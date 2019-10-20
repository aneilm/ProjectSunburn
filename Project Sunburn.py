#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/AM/Desktop/dev/python/Datathon2019/combined.csv")
factor_names = ["carbon_offset_metric_tons", "yearly_sunlight_kwh_kw_threshold_avg", "income"]
factors = df[factor_names]
kmeans = KMeans(n_clusters=20, random_state=0).fit(factors)
# print(kmeans.cluster_centers_)
maxK = 0
indexMax = 0
count = 0
for kmiter in kmeans.cluster_centers_:
    if sum(kmiter) > maxK:
        indexMax = count
        maxK = sum(kmiter)
    count+=1
# print(indexMax)
cluster_map = pd.DataFrame()
cluster_map["data_index"] = df.index.values
cluster_map["cluster"] = kmeans.labels_
# print(cluster_map["cluster"])
# print(len(cluster_map[cluster_map.cluster == indexMax]))
# print(cluster_map[cluster_map.cluster == indexMax])
tempdf = cluster_map[cluster_map.cluster == indexMax]
# print(tempdf)

results = pd.DataFrame()
for i in tempdf.iloc[:, 0]:
    results = results.append(df.iloc[i+2])
df["carbon_n"] = (df["carbon_offset_metric_tons"]-df["carbon_offset_metric_tons"].mean())/df["carbon_offset_metric_tons"].std()
df["sun_n"] = (df["yearly_sunlight_kwh_kw_threshold_avg"]-df["yearly_sunlight_kwh_kw_threshold_avg"].mean())/df["yearly_sunlight_kwh_kw_threshold_avg"].std()
df["income_n"] = (df["income"]-df["income"].mean())/df["income"].std()
df["sum"] = df["carbon_n"]+df["sun_n"]+df["income_n"]

df = df.sort_values("sum", ascending=False)
# print(df)

results_n = df.head(10)
# print(results_n)
coordinates = pd.DataFrame()
coordinates["lat"] = results_n["lat_avg"]
coordinates["lng"] = results_n["lng_avg"]
print("recommended target locations for individual solar panel implementation")
print(coordinates)

results_cik = pd.DataFrame()
results_cik["zip_code"] = results["region_name"]
results_cik["lat_avg"] = results["lat_avg"]
results_cik["lng_avg"] = results["lng_avg"]
results_cik["carbon_offset_metric_tons"] = results["carbon_offset_metric_tons"]
results_cik["income"] = results["income"]
results_cik["yearly_sunlight_kwh_kw_threshold_avg"] = results["yearly_sunlight_kwh_kw_threshold_avg"]
results_cik["sum"] = results_cik["carbon_offset_metric_tons"]+results_cik["income"]+results_cik["yearly_sunlight_kwh_kw_threshold_avg"]
# print(results_cik)
results_cik = results_cik[results_cik["sum"]-results_cik["sum"].mean() >= (0.25*results_cik["sum"].std())]
# print(results_cik)
results_cik = results_cik.sort_values("sum", ascending=False)
# print(results_cik)


# In[ ]:





# In[ ]:




