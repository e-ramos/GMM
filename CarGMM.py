"""IMPORTS"""
## This one is absurd....
from __future__ import division
## WARNING IMPORTING FROM THE FUTURE

#!

__author__ = "E. A. Ramos"
__email__ = "ear@ku.edu"

"""
            Algorithm for Auto MPG Data Sourced at Kaggle
            EECS 738: Machine Learning

            This is an unsupervised one dimensional Gaussian Mixture Model MLE approach using the Expectation Maximization Algorithm and the 
            Bayesian Information Criterion

            Data URL:https://www.kaggle.com/uciml/autompg-dataset

            Github:
            

            Note: Expects data to be extracated in  ~/Datasets/autompg-dataset/auto-mpg.csv' relative to the path of this file. 
"""



import os
import numpy as np
import scipy as sp
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
import seaborn as sns
from scipy.stats import norm

# Directory to store results

script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, 'Results/Cars')
if not os.path.isdir(output_dir):
   os.makedirs(output_dir)



"""Intake Dataset and Remove Categoricals"""

dfmain = pd.read_csv('Datasets/autompg-dataset/auto-mpg.csv')
dfmain = dfmain.drop(['cylinders', 'acceleration', 'model year', 'origin', 'car name'], axis = 1)

#Further Thinning For 2 Variables

df = dfmain.loc[:, dfmain.columns.intersection(['weight','mpg'])]


#Sorting and Separating Data
x1 = np.sort(df["weight"].values)
x2 = np.sort(df["mpg"].values)

## Plotting
ax1 = sns.distplot(x1, bins=50, kde=False)
ax1.set(xlabel='Car Weight', ylabel='Occurences')
plt.title('Histogram of Car Weight')
plt.savefig('{}/InitialDistro.png'.format(output_dir))

#ax2 = sns.distplot(x2, bins=50, kde=False)
#ax2.set(xlabel='Car MPG', ylabel='Occurences')
#plt.title('Histogram of Car Mileage')
#plt.savefig('Results/Images/InitialDistro2.png')

#Getting Data Information

m = len(x1) #Number of Samples

## Result Storage

#############################################
# Search Size for K

K_Search = 10
#############################################

# Storing Means
MusDict = {'For K Gaussians': range(1, 11)}
dfMus = pd.DataFrame(data=MusDict)

#Storing Variance
dfSigs = pd.DataFrame(data=MusDict)

#Storing Weights
dfPis = pd.DataFrame(data=MusDict)

#Storing Score
dfScore = pd.DataFrame(data=MusDict)

# Initializing
initvec = np.empty((K_Search))
initvec[:] = np.nan

for jj in range(1, K_Search+1):
    exec('dfMus[\'Mu k={}\']=initvec'.format(jj))
    exec('dfSigs[\'Sig k={}\']=initvec'.format(jj))
    exec('dfPis[\'Pi k={}\']=initvec'.format(jj))
    exec('dfScore[\'BIC\']=initvec')

"""Iterate Over K 1:10"""
for k in range(K_Search):



    """Initializing EM Algorithm"""

    # Number of Clusters For this iteration
    k = k+1

    # Selecting Random Mean

    mus = np.random.randint(min(x1), max(x1), size=(k)).astype("float")

    # Initialize each Gaussian with Overall variance

    sigmas = np.ones(k)*np.std(x1)

    # Assigning Default Weights to k Distributions

    pis = np.ones(k)/k

    """Plotting Initial"""
    # Settting Colors

    color=iter(plt.cm.rainbow(np.linspace(0,1,k+1)))



    # Histogram
    fig, ax = plt.subplots()
    sns.distplot(x1, bins=50, kde=False, ax=ax, norm_hist = True)

    #Initial Distributions

    totd=np.zeros(len(x1))

    for qq in range(k):
        d0 = pis[qq]*norm.pdf(x1, mus[qq], sigmas[qq])
        totd = totd+d0
        c=next(color)
        plt.plot(x1,d0,c=c, label="k-dist")
        
    #Sum of Initials

    c=next(color)
    plt.plot(x1,totd,c=c, label="Sum Of Dists")
        
        
        

    ax.legend()
    ax.set(xlabel='Car Weight', ylabel='Relative Probabilitty')
    plt.title('Histogram of Car Weight With Random Distribution Guess')
    plt.savefig('{}/InitialGuess.png'.format(output_dir))

    """ITERATING EM ALGO"""

############### HOW MANY ITERATIONS? #################

    steps_to_converge = 501


    for qq in range(steps_to_converge): 
        print("Iteration Index {}".format(qq+1))
        
        """Expectation Stage"""
        priors = np.zeros((m, k)) #Will Hold the Probabilities For m Data Points for k distributions
        
        ## Inner Iteration over k
        
        for jj in range(k):
        
            #PDFs
            priors[:, jj] = norm.pdf(x1, mus[jj], sigmas[jj])
            
        #Weighting by prior probability of a each gaussian then dividing by total weight

        Ri = pis*priors
        TotWeights = 1/np.sum(Ri, axis = 1)
        NewProbs= TotWeights.reshape(m, 1)*Ri
            
            
        """Maximization Stage"""
        
        Oldmus = mus
        for jj in range(k): 
            
            
            
            #For Each cluster calculate mean probability
            pis[jj] = np.mean(NewProbs[:, jj])
            
            #Take Weighted Average of values to find means
            mus[jj] = np.average(x1, weights=(NewProbs[:, jj]))
            
            #Take Weighted Average of Variances
            secmoment = np.square(x1-mus[jj])
            
            
            sqrsig = np.average(secmoment, weights=NewProbs[:, jj])
            
            sigmas[jj]= np.sqrt(sqrsig)

        
        """Plotting Results"""
    # Settting Colors

    color=iter(plt.cm.jet(np.linspace(0,1,k+1)))

    # Histogram
    fig, ax = plt.subplots()
    sns.distplot(x1, bins=50, kde=False, ax=ax, norm_hist = True)

    #Final Distributions

    totd=np.zeros(len(x1))

    for qq in range(k):
        d0 = pis[qq]*norm.pdf(x1, mus[qq], sigmas[qq])
        totd = totd+d0
        c=next(color)
        plt.plot(x1,d0,c=c, label="k-dist")
            
    #Sum of Finals

    c=next(color)
    plt.plot(x1,totd,c=c, label="Sum Of Dists")
    
    ax.legend()
    ax.set(xlabel='Car Weight', ylabel='Relative Probabilitty')
    plt.title('Histogram of Car Weight After EM Algo')
    plt.savefig('{}/Final{}.png'.format(output_dir,k))

    plt.close("all")

    """Calculating BIC"""

    """First Calculate Square Error"""

    KDist = totd/np.sum(totd) #normalizing

    #getting Relative Frequency from samples
    relfreq , binedges = np.histogram(x1, bins=50)
    relfreq = relfreq/np.sum(relfreq)

    binn=(binedges[1:] + binedges[:-1]) / 2

    #Set Search range and normalize bins to get appropriate indexes from 
    #KDist
    high =max(x1)
    low = min(x1)
    span = high-low

    #Indexes to use:
    binn = np.round((len(x1)*(binn-low))/(span)).astype(int) 

    ## Residual Sum of Squares (SSE)

    SSE = np.sum(np.square(KDist[binn]-relfreq))

    """BIC"""
    #BIC for Gaussian Special Case:
    # is k*ln(n)+n*(ln(SSE/n) Where n is the number of observations
    nn = len(x1)
    
    BIC = k*np.log(nn)+(nn*(np.log(SSE/nn)))
    
    """Storage"""

    for jj in range(k):
        exec('dfMus.at[{}, \'Mu k={}\']=mus[{}]'.format(jj, k, jj))
        exec('dfSigs.at[{}, \'Sig k={}\']=sigmas[{}]'.format(jj, k, jj))
        exec('dfPis.at[{}, \'Pi k={}\']=pis[{}]'.format(jj, k, jj))
        exec('dfScore.at[{}, \'BIC\']=BIC'.format(k-1))

        os.system('cls')
        
        
    print(dfScore)
#Outputting to CSV

## Plotting
plt.plot(np.arange(1, K_Search+1), dfScore['BIC'])
plt.xlabel('Number Of Gaussians')
plt.ylabel('BIC')
plt.title('Bayes Information Criterion vs. Gaussians')
plt.savefig('{}/BIC.png'.format(output_dir))
plt.show()

dfMus.to_csv('{}/Mus.csv'.format(output_dir), index = None, header=True)
dfSigs.to_csv('{}/Sigs.csv'.format(output_dir), index = None, header=True)
dfPis.to_csv('{}/Pis.csv'.format(output_dir), index = None, header=True)
dfScore.to_csv('{}/Score.csv'.format(output_dir), index = None, header=True)

"""Finding Best Model and Outputting Results"""
bestk = np.argmax(np.abs(np.diff(np.diff(dfScore['BIC']))))+2

print('Best BIC found at k = {}'.format(bestk))
print('Plotting')

img=mpimg.imread('{}/Final{}.png'.format(output_dir,bestk))
imgplot = plt.imshow(img)
plt.show()