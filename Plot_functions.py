#Plot_functions.py

##########################################################################################################
#                                        Import librairies                                               #
##########################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import scipy.io as sio
from scipy.stats import norm
import scipy.stats
import itertools
import math

import pcalg
import cdt
from cdt.metrics import SHD
from cdt.metrics import SID

import pgmpy
from gsq.ci_tests import ci_test_bin, ci_test_dis

from sklearn import preprocessing
import os

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import ward, fcluster

from Structure_Learning import *


##########################################################################################################
#                                           Progress bar                                                 #
##########################################################################################################
def progressBar(current, total,barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

    
##########################################################################################################
#                                        Plot dendrogram                                                 #
##########################################################################################################

def Plot_dendrogram(distance_matrix,methode,title,size_x,size_y):
    # compute linkage with alternative method
    X = distance_matrix
    Z = linkage(X, method=methode)
    
    # plot dendrogram
    fig = plt.figure(figsize=(size_x,size_y))
    plt.rc('font', family='serif')
    dn = dendrogram(Z,color_threshold=0.7*max(Z[:,2]))
    
    # add title, axis labels, ...
    plt.title(title,fontsize=25)
    plt.xlabel("Store index.",fontsize=25)
    plt.xticks(fontsize=16)
    plt.ylabel('Distance in SHD between cluster',fontsize=25)
    plt.yticks(fontsize=20)
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    plt.show()

##########################################################################################################
#                      Plot Boxplot for hyper parameter optimization                                     #
##########################################################################################################

def Box_plot(distances,name,Hyperarameter,random_dist):
    # create a dataframe and pivot it
    df = pd.DataFrame([distances,name]).T
    df.columns = ['Mean',Hyperarameter]
    df = df.pivot_table(values='Mean', index=df.index, columns= Hyperarameter, aggfunc='first')
    
    # generate variance in order to obtain eliptic representation rather than a line
    vals, names, xs = [],[],[]
    plt.figure(figsize=(25, 10))
    plt.rc('font', family='serif')
    for i, col in enumerate(df.columns):
        vals.append(df[col].values)
        names.append(col)
        xs.append(np.random.normal(i + 1, 0.02, df[col].values.shape[0]))
    
    # plot the box plot 
    df.boxplot(grid=False, fontsize=15)
    palette = ['lightcoral', 'lightgreen', 'lightskyblue','gold','paleturquoise',
               'coral','wheat','orange','lightsteelblue']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)

    plt.axhline(y=random_dist, color='#ff3300', linestyle='--', linewidth=1,  label='Random level')
    plt.title('SHD distances repartition function of ' + Hyperarameter,fontsize=25)
    plt.ylabel('Distance in SHD',fontsize=25)
    plt.xlabel(Hyperarameter,fontsize=25)
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    plt.show()
    
##########################################################################################################
#                            Display central network of each cluster                                     #
##########################################################################################################
    
def Plot_Clusters_Network(X,n_cluster,link,skeleton,h_x,h_y):
    
    # create clusters with alternative approach
    Z = linkage(X, method=link)
    labels = fcluster(Z,n_cluster,criterion='maxclust')
    
    # create empty list for storage
    distances = []
    cluster = []
    
    # input parameters for subplot 
    b = 2  # number of columns
    a = math.ceil(len(np.unique(labels)) / b)  # number of rows
    c = 0  # initialize plot counter
    plt.rc('font', family='serif')
    fig = plt.subplots(figsize=(h_x,h_y))
    
    # iterate on distinct labels
    for k in np.unique(labels):
        g_list = []
        s = 0
        
        # group skeletons by labels 
        for i in range(len(skeleton)):
            if labels[i] == k:
                s += 1
                
                g_list.append(skeleton[i])
        # compute the distance matrix within each cluster
        shd_dist = np.zeros((s,s))
        for i in range(s):
            for j in range(i,s):
                shd_dist[i,j] = SHD(g_list[i],g_list[j])
                shd_dist[j,i] = SHD(g_list[j],g_list[i])
                
        c += 1
        fig = plt.subplot(a,b,c)
        # display central graph for each cluster
        fig = nx.draw_circular(g_list[np.argmin(np.mean(shd_dist, axis=0))],
                               arrowsize=20, with_labels=True, font_weight='bold',
                               width=0.2,node_color='lightsteelblue')
        fig = plt.title('Central graph in cluster: ' + str(k),fontsize=15)
        
    plt.show()

    
##########################################################################################################
#                               Plot intra variance cluster evolution                                    #
##########################################################################################################

def Plot_Variance_Evolution(dist,n_c,skeleton,methode,title,size_x,size_y):
    
    # initialization for storage
    Variation_Conv = []
    Variation_Alt = []
    N = []
    
    count = 0
    for n in range(1,n_c+1):
        count += 1
        # computation for conventionnal method
        dist_Conv,cluster_Conv = Predict_labels_Conventionnal(dist,n,skeleton,methode=methode)
        D_Conv = pd.DataFrame([cluster_Conv,np.array(dist_Conv)]).T
        D_Conv.columns = ['Cluster','Distance']
        D_Conv['Distance'] = D_Conv['Distance'].astype(float)
        D_Conv = D_Conv.groupby(['Cluster']).var()
        
        # computation for alternative method
        count += 1
        dist_Alt,cluster_Alt = Predict_labels_Alt(dist,n,skeleton,methode=methode)
        D_Alt = pd.DataFrame([cluster_Alt,np.array(dist_Alt)]).T
        D_Alt.columns = ['Cluster','Distance']
        D_Alt['Distance'] = D_Alt['Distance'].astype(float)
        D_Alt = D_Alt.groupby(['Cluster']).var()

        # store results
        Variation_Conv.append(np.mean(D_Conv))
        Variation_Alt.append(np.mean(D_Alt))
        N.append(n)
    
    #  define graph size and style  
    plt.figure(figsize=(size_x,size_y))
    plt.rc('font', family='serif')
    
    # define plot variable, title, color, ...
    plt.scatter(N,Variation_Alt,s=150, facecolors='none', marker='^',edgecolors='black',label='Alternative approach')
    plt.plot(N, Variation_Alt, '--', color='gray')
    plt.scatter(N,Variation_Conv,s=150, facecolors='none', edgecolors='black',label='Conventionnal approach')
    plt.plot(N, Variation_Conv, '--', color='gray')
    plt.title(title,fontsize=22)
    plt.xlabel('Number of cluster',fontsize=18)
    plt.ylabel('Average intra cluster variance',fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    plt.show()
    
    
##########################################################################################################
#                               Plot intra variance cluster evolution                                    #
##########################################################################################################

def Plot_Variance(dist,n_c,skeleton,methode,title,size_x,size_y):
    
    # initialization for storage
    Variation_Alt = []
    N = []
    
    count = 0
    for n in range(1,n_c+1):
        count += 1
        
        # computation for alternative method
        count += 1
        dist_Alt,cluster_Alt = Predict_labels_Alt(dist,n,skeleton,methode=methode)
        D_Alt = pd.DataFrame([cluster_Alt,np.array(dist_Alt)]).T
        D_Alt.columns = ['Cluster','Distance']
        D_Alt['Distance'] = D_Alt['Distance'].astype(float)
        D_Alt = D_Alt.groupby(['Cluster']).var()

        # store results
        Variation_Alt.append(np.mean(D_Alt))
        N.append(n)
    
    #  define graph size and style  
    plt.figure(figsize=(size_x,size_y))
    plt.rc('font', family='serif')
    
    # define plot variable, title, color, ...
    plt.scatter(N,Variation_Alt,s=150, facecolors='none', marker='^',edgecolors='black',label='Alternative approach')
    plt.plot(N, Variation_Alt, '--', color='gray')
    plt.title(title,fontsize=22)
    plt.xlabel('Number of cluster',fontsize=18)
    plt.ylabel('Average intra cluster variance',fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    plt.show()