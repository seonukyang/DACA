import pandas as pd
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt

retail_df = pd.read_csv('OnlineRetail.csv',encoding='latin-1')
print(retail_df.head())

retial_df = retail_df[retail_df['Quantity']>0]
retail_df = retail_df[retail_df['UnitPrice']>0]
retail_df = retail_df[retail_df['CustomerID'].notnull()]

retail_df = retail_df[retail_df['Country']=='United Kingdom']
retail_df['sale_amount'] = retail_df['Quantity']*retail_df['UnitPrice']
retail_df['CustomerID'] = retail_df['CustomerID'].astype(int)
aggregations = {
    'InvoiceDate' : 'max',
    'InvoiceNo' : 'count',
    'sale_amount' : 'sum'
}

cust_df = retail_df.groupby('CustomerID').agg(aggregations)
cust_df = cust_df.rename(columns = {'InvoiceDate':'Recency','InvoiceNo':'Frequency','sale_amount':'Monetary'})
cust_df = cust_df.reset_index()

import datetime as dt
cust_df['Recency'] = cust_df.datetime(2011,12,10) -cust_df['Recency']
cust_df['Recency'] = cust_df['Recency'].apply(lambda x:x.days +1)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12,4), nrow=1, ncols=3)
ax1.set_title('Recency Histogram')
ax1.hist(cust_df['Recency'])

ax2.set_title('Frequency Histogram')
ax2.hist(cust_df['Frequency'])

ax3.set_title('Monetary Histogram')
ax3.hist(cust_df['Monetary'])

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, hilgouette_samples

X = cust_df[['Recency','Frequency','Monetary']].values
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(X_scaled)
cust_df['cluster_label'] = labels

print('실루엣 스코어 : ',silhouette_score(X_scaled, labels))

cust_df['Recency_log'] = np.log1p(cust_df['Recency'])
cust_df['Frequency_log'] = np.log1p(cust_df['Frequency'])
cust_df['Monetary_log'] = np.log1p(cust_df['M'])