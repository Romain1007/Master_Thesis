#Structure_Learning.py

##########################################################################################################
#                                           Import librairies                                            #
##########################################################################################################
import numpy as np
import pandas as pd 

from datetime import date
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import scipy.io as sio
from scipy.stats import norm
import scipy.stats
import itertools 

import pcalg
import cdt
from cdt.metrics import SHD

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram, linkage

from pgmpy.estimators.CITests import pearsonr
from fcit import fcit


##########################################################################################################
#                                           Data preprocessing                                           #
##########################################################################################################

def data_formating(data):
    # select subpart of the date given in data
    data['date'] = [data['date'][i][:18] for i in data.index]
    # pivot table in order to create lines with all measures
    formated = pd.pivot_table(data, values=['duration'], index=['date'],
                    columns=['operation'], aggfunc= np.median,fill_value=0)
    # preprocess columns names
    operation = formated.columns
    operation = [op[1] for op in operation]
    # transform date in datetime object
    formated.index = pd.to_datetime(formated.index)
    # group observation by range of time
    formated = formated.groupby(pd.Grouper(freq='265 S')).mean()
    formated = formated.dropna(axis=0)
    formated = np.array(formated)
    
    return formated, operation


##########################################################################################################
#                                    Country structural discovery                                        #
##########################################################################################################

def Country_structure(country,a,op,independence_test):
    # import data
    data = pd.read_csv(country + '.csv')
    # choose measures of interest
    mask = [data['operation'][i] in op for i in data.index] 
    data = data[mask]
    # exclude web measures
    data = data[['WEB' not in data['store_code'][i] for i in data.index]]
    # create a list of distinct store
    stores = data['store_code'].unique()
    analysis_store = []
    
    # run inference on each store
    skeletons = []
    count = 0
    for s in stores:
        # preprocess data
        POS, columns = data_formating(data[data['store_code'] == s])
        pd_POS = pd.DataFrame(POS)
        pd_POS.columns = columns
        label = {i : columns[i] for i in range(len(columns))}   
        
        # select stores with enough observations
        if POS.shape[0] > 100:
            analysis_store.append(s)
            count += 1
            # estimate the skeleton
            (g, sep_set) = pcalg.estimate_skeleton(indep_test_func=independence_test,data_matrix = POS,alpha = a)
            # estimate the directed graph
            h = pcalg.estimate_cpdag(g, sep_set)
            # add measure name to nodes
            h = nx.relabel_nodes(h, label)
            skeletons.append(h)   
            
    # Compute distance matrix
    shd_dist = np.zeros((len(skeletons),len(skeletons)))
    for i in range(len(skeletons)):
        for j in range(len(skeletons)):
            shd_dist[i,j] = SHD(skeletons[i],skeletons[j])
    
    return shd_dist,skeletons, analysis_store


##########################################################################################################
#                            Clustering with conventionnal approach                                      #
##########################################################################################################

def Predict_labels_Conventionnal(dist,n,skeletons,methode):
    '''
     This function assign cluster to observations based on agglomerative clustering,
     using distance matrix to perform clustering (conventinnal approach)
    '''
    # create the clustering
    X = dist     
    labels = AgglomerativeClustering(n_clusters=n,affinity='precomputed',linkage=methode).fit_predict(X)

    distances = []
    cluster = []
    # iterate through distinct labels
    for k in np.unique(labels):
        g_list = []
        s = 0
        for i in range(len(skeletons)):
            if labels[i] == k:
                s += 1
                g_list.append(skeletons[i])

        shd_dist = np.zeros((s,s))
        # compute distance matrix within each cluster
        for i in range(s):
            for j in range(i,s):
                shd_dist[i,j] = SHD(g_list[i],g_list[j])
                shd_dist[j,i] = SHD(g_list[j],g_list[i])

        iu1 = np.triu_indices(shd_dist.shape[0],1)
        
        # return assigned cluster and distances between observation in the given cluster
        distances = distances + list(shd_dist[iu1])
        cluster = cluster + ['Cluster: ' + str(k) + ' (' + str(len(g_list)) +')' for l in range(len(list(shd_dist[iu1])))]

    return distances,cluster


##########################################################################################################
#                              Clustering with alternative approach                                      #
##########################################################################################################

def Predict_labels_Alt(dist,n,skeletons,methode):
  
    # alternative clustering
    Z = linkage(dist, method=methode)
    labels = fcluster(Z,n,criterion='maxclust')

    distances = []
    cluster = []
    # iterate through distinct labels
    for k in np.unique(labels):
        g_list = []
        s = 0
        for i in range(len(skeletons)):
            if labels[i] == k:
                s += 1
                g_list.append(skeletons[i])

        shd_dist = np.zeros((s,s))
        # compute distance matrix within each cluster
        for i in range(s):
            for j in range(i,s):
                shd_dist[i,j] = SHD(g_list[i],g_list[j])
                shd_dist[j,i] = SHD(g_list[j],g_list[i])

        iu1 = np.triu_indices(shd_dist.shape[0],1)
        # return assigned cluster and distances between observation in the given cluster
        distances = distances + list(shd_dist[iu1])
        cluster = cluster + ['Cluster: ' + str(k) + ' (' + str(len(g_list)) +')' for l in range(len(list(shd_dist[iu1])))]

    return distances,cluster