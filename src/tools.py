#Utility
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#RFM
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
from scipy.stats import kruskal

#



def rfm(df):
    """Create unique user dataframe"""
    
    user = pd.DataFrame(df['customer id'].unique())
    user.columns = ['customer id']
    
    """defines customer recency"""
    
    max_purchase = df.groupby(['customer id'])['invoicedate'].max()
    max_purchase = pd.DataFrame(max_purchase).reset_index()
    max_purchase.columns = ['customer id', 'maxPurchaseDate']
    max_purchase['recency'] = (max_purchase.maxPurchaseDate.max()-max_purchase.maxPurchaseDate).dt.days
    
    """defines customer frequency"""
    
    frequency = df.groupby(['customer id'])['invoicedate'].count().reset_index()
    frequency.columns = ['customer id', 'frequency']
    
    """defines customer monetary"""
    
    monetary = df.groupby(['customer id'])['revenue'].sum().reset_index()
    monetary.columns = ['customer id', 'monetary']
    
    """merges user RFM into one dataframe"""
    
    user = pd.merge(user, max_purchase[['customer id', 'recency']], on='customer id')
    user = pd.merge(user, frequency, on='customer id')
    user = pd.merge(user, monetary, on='customer id')
    
    return user


def rfmclusters(df):

    """show elbow chart for recency,
    frequency and monetary of a dataframe to help 
    determine the number of clusters (K)"""

    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle(f'Clusters')

    model = KMeans()

    visualizer = KElbowVisualizer(model, k=(1,10), ax = ax1)
    visualizer.fit(df[['recency']])

    visualizer = KElbowVisualizer(model, k=(1,10), ax = ax2)
    visualizer.fit(df[['frequency']])

    visualizer = KElbowVisualizer(model, k=(1,10), ax = ax3)
    visualizer.fit(df[['monetary']])

    return plt.show()


def clustering (df, k):
    """
    returns a cluster per each user,
    per category: recency, frequency, monetary.
    
    """
    kmeans = KMeans(n_clusters=k)
    
    kmeans.fit(df[['recency']])
    df['r_cluster'] = kmeans.predict(df[['recency']])
    
    kmeans.fit(df[['frequency']])
    df['f_cluster'] = kmeans.predict(df[['frequency']])
    
    kmeans.fit(df[['monetary']])
    df['m_cluster'] = kmeans.predict(df[['monetary']])
    
    return df



def score (df):
    """assign a score to a user based on their rfm values
    """

    df['score'] = df['r_cluster'] + df['f_cluster'] + df['m_cluster']
    return df 


def segmentation(df):
    """
    assign a category to a customer based on their score
    """ 
    
    df['segment'] = 'Low-Value'
    df.loc[df['score'] >= 1,'segment'] = 'Mid-Value' 
    df.loc[df['score'] >= 4,'segment'] = 'High-Value' 
    return df
    