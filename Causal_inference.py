# Causal_inference.py

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


##########################################################################################################
#                                  Conditionnal independance test                                        #
##########################################################################################################

def CItest(D,X,Y,Z):
    alpha = 0.05
    n = D.shape[0]
    c = norm.ppf(1-alpha/2)
    columns = [X,Y] + Z
    DD = D[:, columns]
    R = np.corrcoef(DD.T)
    P = np.linalg.inv(R)
    ro = -P[0,1] / np.sqrt(P[0,0] * P[1,1])
    zro = 0.5 * np.log((1+ro)/(1-ro))
    
    if np.abs(zro) < c / np.sqrt(n-len(Z)-3.0):
        CI = 1
    else:
        CI = 0
    
    return CI


##########################################################################################################
#                                           PC algorithm                                                 #
##########################################################################################################

def PC(D,columns):
    # Store skeletons & v-strucutre during iterations
    skeletons = []
    v_struct = []
    # Create an fully connected graph
    E = np.ones((D.shape[1],D.shape[1]))
    np.fill_diagonal(E, 0)
    G = nx.Graph(E)
    label = {i : columns[i] for i in range(len(columns))}  
    G = nx.relabel_nodes(G, label)
    
    # Iterate on the number of variable in conditionning set
    for n in range(D.shape[1]):
        progressBar(n, D.shape[1]-1)
        skeletons.append(nx.adjacency_matrix(G))
        # Iterate on each combination of 2 distinct node in network
        for i in range(D.shape[1]):
            for j in range(i,D.shape[1]):
                if i != j: 
                    # Check if there is a link in graph beteween the 2 nodes
                    if G.has_edge(columns[i], columns[j]):
                        Z = list(np.arange(0,D.shape[1])) # list of others nodes
                        Z.remove(i) # without current nodes
                        Z.remove(j) # without current nodes
                        Z_combinations = list(itertools.combinations(Z, n)) # check all combinations
                        # Perform conditionnal independance for each set Z 
                        ci_test = [CItest(D,i,j,list(z)) for z in Z_combinations] 
                        
                        # If the variables are independant for all conditionning set
                        if np.sum(ci_test) != 0:
                            G.remove_edge(columns[i], columns[j])
                            # Store v-structure idenfy at iteration 2 (n=1)
                            if n == 1:
                                v_struct = v_struct + [(columns[i], columns[j],Z_combinations[l]) 
                                                       for l in range(len(ci_test))
                                                       if ci_test[l] != 0]
                                   
    # Adjacency matrix
    A = nx.adjacency_matrix(G)
    
    return A.todense(), skeletons, v_struct


##########################################################################################################
#                                           Skeleton orientation                                         #
##########################################################################################################

def edge_orientation(v_struct,A):
    # create directed graph
    G = nx.DiGraph(A)
    
    # Orient v-structures
    for v in v_struct: # iterate on all v-structure identified by PC algorithm
        X, Y, Z = v[0],v[1],v[2][0]
        # check if there is nodes presenting a v-structure skeleton
        mask_X =  G.has_edge(X, Z) & (G.has_edge(X, Y)== False)
        mask_Y =  G.has_edge(Y, Z) & (G.has_edge(Y, X)== False)
        mask_Z =  G.has_edge(Z, Y) & G.has_edge(Z, X)
        #if  G.has_edge(Z, X) & G.has_edge(Z, Y) & G.has_edge(X, Z) & G.has_edge(Y, Z) :
        if  mask_X & mask_Y & mask_Z:
            # if yes, remove edge to create a v-structure
            G.remove_edge(Z,X)
            G.remove_edge(Z,Y)
            
    # Rule 1
    for l in list(itertools.permutations(list(G.nodes), 3)): # iterate on all triple combination
        X, Y, Z = l[0],l[1],l[2]
        # select triple which are satisfying the rule 1 condition
        mask_X =  G.has_edge(X, Y) & (G.has_edge(X, Z) == False)
        mask_Y =  G.has_edge(Y, Z) & (G.has_edge(Y, X)== False)
        mask_Z =  G.has_edge(Z, Y) & (G.has_edge(Z, X)== False)
        # if it satifies the condition, orient the structure
        if  mask_X & mask_Y & mask_Z:
            G.remove_edge(Z,Y)
            
    # Rule 2
    for l in list(itertools.permutations(list(G.nodes), 4)): # iterate on all quadruple combination
        X, Y, Z, W = l[0], l[1], l[2], l[3]
        # select quadruple which are satisfying the rule 2 condition
        mask_X =  G.has_edge(X, Y) & G.has_edge(X, Z) 
        mask_Y =  G.has_edge(Y, Z) & (G.has_edge(Y, X)==False)
        mask_Z =  G.has_edge(Z, X) & (G.has_edge(Z, Y)==False)
        # if it satifies the condition, orient the structure
        if  mask_X & mask_Y & mask_Z:
            G.remove_edge(Z,X)
            
    # Rule 3
    for l in list(itertools.permutations(list(G.nodes), 4)): # iterate on all quadruple combination
        X, Y, Z, W = l[0], l[1], l[2], l[3]
        # select triple which are satisfying the rule 3 condition
        mask_X =  G.has_edge(X, Y) & G.has_edge(X, Z) & G.has_edge(X, W)
        mask_Y =  G.has_edge(Y, X) & G.has_edge(Y, Z) & (G.has_edge(Y, W)==False)
        mask_Z =  G.has_edge(Z, X) & (G.has_edge(Z, Y)==False)&(G.has_edge(Z, W)==False)
        mask_W =  G.has_edge(W, X) & G.has_edge(W, Z) & (G.has_edge(W, Y)==False)
        # if it satifies the condition, orient the structure
        if  mask_X & mask_Y & mask_Z & mask_W:
            G.remove_edge(Z,X)
            
    # Rule 4
    for l in list(itertools.permutations(list(G.nodes), 4)): # iterate on all quadruple combination
        X, Y, Z, W = l[0], l[1], l[2], l[3]
        # select triple which are satisfying the rule 4 condition
        mask_X =  G.has_edge(X, Y) & G.has_edge(X, Z) & G.has_edge(X, W)
        mask_Y =  G.has_edge(Y, X) & G.has_edge(Y, W) & (G.has_edge(Y, Z)==False)
        mask_Z =  G.has_edge(Z, X) & (G.has_edge(Z, Y)==False)&(G.has_edge(Z, W)==False)
        mask_W =  G.has_edge(W, X) & G.has_edge(W, Z) & (G.has_edge(W, Y)==False)
        # if it satifies the condition, orient the structure
        if  mask_X & mask_Y & mask_Z & mask_W:
            G.remove_edge(X,Z)

    return G


##########################################################################################################
#                                      Show skeleton orientation steps                                   #
##########################################################################################################

def edge_orientation_evolution(v_struct,A):
    # create directed graph
    G = nx.DiGraph(A)
    
    b = 3  # number of rows
    a = 2  # number of columns
    
    fig = plt.figure(figsize=(20,10))
    plt.rc('font', family='serif')

    
    plt.subplot(a, b, 1)
    plt.title('Graph before orientation')
    nx.draw_circular(G, with_labels=True,arrowsize=20, font_weight='bold',width=0.2,node_color='lightsteelblue')
    
    # orient v-structures
    n = 0
    for v in v_struct: # iterate on all v-structure identified by PC algorithm
        X = v[0]
        Y = v[1]
        Z = v[2][0]
        # check if there is nodes presenting a v-structure skeleton
        mask_X =  G.has_edge(X, Z) & (G.has_edge(X, Y)== False)
        mask_Y =  G.has_edge(Y, Z) & (G.has_edge(Y, X)== False)
        mask_Z =  G.has_edge(Z, Y) & G.has_edge(Z, X)
        #if  G.has_edge(Z, X) & G.has_edge(Z, Y) & G.has_edge(X, Z) & G.has_edge(Y, Z) :
        if  mask_X & mask_Y & mask_Z:
            # if yes, remove edge to create a v-structure
            G.remove_edge(Z,X)
            G.remove_edge(Z,Y)
            n += 2
            
    plt.subplot(a, b, 2)
    plt.title('V-structures orientation (' + str(n) + ' orientations performed)')
    nx.draw_circular(G, with_labels=True,arrowsize=20, font_weight='bold',width=0.2,node_color='lightsteelblue')
    
    # Rule 1
    n = 0
    for l in list(itertools.permutations(list(G.nodes), 3)): # iterate on all triple combination
        X = l[0]
        Y = l[1]
        Z = l[2]
        # select triple which are satisfying the rule 1 condition
        mask_X =  G.has_edge(X, Y) & (G.has_edge(X, Z) == False)
        mask_Y =  G.has_edge(Y, Z) & (G.has_edge(Y, X)== False)
        mask_Z =  G.has_edge(Z, Y) & (G.has_edge(Z, X)== False)
        # if it satifies the condition, orient the structure
        if  mask_X & mask_Y & mask_Z:
            G.remove_edge(Z,Y)
            n += 1
    plt.subplot(a, b, 3)
    plt.title('Rule 1 orientation  (' + str(n) + ' orientations performed)')
    nx.draw_circular(G, with_labels=True,arrowsize=20, font_weight='bold',width=0.2,node_color='lightsteelblue')
    
    
    # Rule 2
    n = 0
    for l in list(itertools.permutations(list(G.nodes), 4)): # iterate on all quadruple combination
        X = l[0]
        Y = l[1]
        Z = l[2]
        W = l[3]
        # select quadruple which are satisfying the rule 2 condition
        mask_X =  G.has_edge(X, Y) & G.has_edge(X, Z) 
        mask_Y =  G.has_edge(Y, Z) & (G.has_edge(Y, X)==False)
        mask_Z =  G.has_edge(Z, X) & (G.has_edge(Z, Y)==False)
        # if it satifies the condition, orient the structure
        if  mask_X & mask_Y & mask_Z:
            G.remove_edge(Z,X)
            n += 1
    plt.subplot(a, b, 4)
    plt.title('Rule 2 orientation  (' + str(n) + ' orientations performed)')
    nx.draw_circular(G, with_labels=True,arrowsize=20, font_weight='bold',width=0.2,node_color='lightsteelblue')

    # Rule 3
    n = 0
    for l in list(itertools.permutations(list(G.nodes), 4)): # iterate on all quadruple combination
        X = l[0]
        Y = l[1]
        Z = l[2]
        W = l[3]
        # select triple which are satisfying the rule 3 condition
        mask_X =  G.has_edge(X, Y) & G.has_edge(X, Z) & G.has_edge(X, W)
        mask_Y =  G.has_edge(Y, X) & G.has_edge(Y, Z) & (G.has_edge(Y, W)==False)
        mask_Z =  G.has_edge(Z, X) & (G.has_edge(Z, Y)==False) & (G.has_edge(Z, W)==False)
        mask_W =  G.has_edge(W, X) & G.has_edge(W, Z) & (G.has_edge(W, Y)==False)
        # if it satifies the condition, orient the structure
        if  mask_X & mask_Y & mask_Z & mask_W:
            G.remove_edge(Z,X)
            n += 1
    plt.subplot(a, b, 5)
    plt.title('Rule 3 orientation  (' + str(n) + ' orientations performed)')
    nx.draw_circular(G, with_labels=True,arrowsize=20, font_weight='bold',width=0.2,node_color='lightsteelblue')

    # Rule 4
    n = 0
    for l in list(itertools.permutations(list(G.nodes), 4)): # iterate on all quadruple combination
        X = l[0]
        Y = l[1]
        Z = l[2]
        W = l[3]
        # select triple which are satisfying the rule 4 condition
        mask_X =  G.has_edge(X, Y) & G.has_edge(X, Z) & G.has_edge(X, W)
        mask_Y =  G.has_edge(Y, X) & G.has_edge(Y, W) & (G.has_edge(Y, Z)==False)
        mask_Z =  G.has_edge(Z, X) & (G.has_edge(Z, Y)==False) & (G.has_edge(Z, W)==False)
        mask_W =  G.has_edge(W, X) & G.has_edge(W, Z) & (G.has_edge(W, Y)==False)
        # if it satifies the condition, orient the structure
        if  mask_X & mask_Y & mask_Z & mask_W:
            G.remove_edge(X,Z)
            n += 1
    plt.subplot(a, b, 6)
    plt.title('Rule 4 orientation  (' + str(n) + ' orientations performed)')
    nx.draw_circular(G, with_labels=True,arrowsize=20, font_weight='bold',width=0.2,node_color='lightsteelblue')
    
    plt.show()
    
    return G

##########################################################################################################
#                                             Progress bar                                               #
##########################################################################################################

def progressBar(current, total,barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')