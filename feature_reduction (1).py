import numpy as np 

class FeatureReduction():
    
    def __init__(self):
        self.model_params = {}
    
    def fit(self, data):
        pass
    
    def predict(self, data):
        pass
    

class PrincipleComponentAnalysis(FeatureReduction):
    '''self.model_params is where you will save your principle components (up to LoV)'''
    ''' Its useful to use a projection matrix as your only param'''
    
    def fit(self, data, thresh=0.95, plot_var = True):
        '''Find the principle components of your data'''
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        data = (data - self.mean)/self.std
        cov = np.cov(data, rowvar=False)
        evalues, evectors = np.linalg.eigh(cov)
        sortedval = np.argsort(evalues)[::-1]
        evalues = evalues[sortedval]
        evectors = evectors[:, sortedval]
        for i in range(evectors.shape[1]):
            if evectors[0, i] < 0:
                evectors[:, i] = -evectors[:, i]
        variance = self.calc_variance(evalues)
        totvar = 0
        num = 0
        for i, var in enumerate(variance):
            totvar += var
            if totvar >= thresh:
                num = i + 1
                break
        self.model_params['projection_matrix'] = evectors[:, :num]
        self.model_params['variance'] = variance[:num]
        
    def predict(self, data):
        ''' You can change this function if you want'''
        data = (data - self.mean) / self.std
        return np.dot(data, self.model_params['projection_matrix'])
    
    def calc_variance(self, evalues):
        '''Input: list of eigen values
           Output: list of normalized values corresponding to percentage of information an eigen value contains'''
        '''Your code here'''
        
        '''Stop Coding here'''
        return evalues / np.sum(evalues)