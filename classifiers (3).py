''' Import Libraries'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class Classifier:
    ''' This is a class prototype for any classifier. It contains two empty methods: predict, fit'''
    def __init__(self):
        self.model_params = {}
        pass
    
    def predict(self, x):
        '''This method takes in x (numpy array) and returns a prediction y'''
        raise NotImplementedError
    
    def fit(self, dataframe):
        '''This method is used for fitting a model to data: x, y'''
        raise NotImplementedError
        
        
        
class KMeans(Classifier):
    '''No init function, as we inherit it from the base class'''
    def fit(self, data, k=2, tol = 0.01):
        '''k is the number of clusters, tol is our tolerance level'''
        '''Randomly choose k vectors from our data'''
        '''Your code here'''
        np.random.seed(42)
        self.centroids = data[np.random.choice(data.shape[0], k, replace=False)]
        pcentroid = np.zeros_like(self.centroids)
        while np.linalg.norm(self.centroids - pcentroid) >= tol:
            pcentroid = self.centroids.copy()
            clusters = [[] for _ in range(k)]
            for point in data:
                distances = [self.calc_distance(point, centroid) for centroid in self.centroids]
                cid = np.argmin(distances)
                clusters[cid].append(point)
            self.centroids = np.array([np.mean(cluster, axis=0) if cluster else pcentroid[i] for i, cluster in enumerate(clusters)])
        
    def predict(self, x):
        '''Input: a vector (x) to classify
           Output: an integer (classification) corresponding to the closest cluster
           Idea: you measure the distance (calc_distance) of the input to 
           each cluster centroid and return the closest cluster index'''
        '''Your code here'''
        distances = [self.calc_distance(x, centroid) for centroid in self.centroids]
        return np.argmin(distances)
    
    def calc_distance(self, point1, point2):
        '''Your code here'''
        '''Input: two vectors (point1 and point2)
           Output: a single value corresponding to the euclidan distance betwee the two vectors'''
        '''Your code here'''
        return np.linalg.norm(point1 - point2)
        
        
