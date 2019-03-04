from sklearn.base import BaseEstimator,RegressorMixin,TransformerMixin
from sklearn.pipeline import Pipeline
from functools import partial
import scipy as sp
import numpy as np
from sklearn.metrics import cohen_kappa_score


        
        
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ =self.models#= [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   
    
    

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p,weights='quadratic')
        #print(ll)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']

def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)


class Pipeout(Pipeline):
    def __init__(self, rates, steps, mode='class', memory=None):
        super().__init__(steps, memory)
        self.wrk_mode = mode
        self.rates = rates
        #self.opt = opt
         
    def to_bins(self, x, borders):
        for i in range(len(borders)):
            if x <= borders[i]:
                return i
        return len(borders)

    def predict(self,*args,**kwargs):
        yh = super(Pipeout,self).predict(*args,**kwargs)
        
        if self.wrk_mode == 'regr':
            return yh
        
        elif self.wrk_mode == 'cool':
            l = len(yh)
            d = dict(zip(range(l),yh))
            sorted_by_value = sorted(d.items(), key=lambda kv: kv[1])
            thrs = []
            cnt = 0
            for i,k in enumerate(sorted_by_value):
                if cnt<5 and i/l > self.rates[cnt]:
                    cnt+=1
                    thrs.append(k[1])
            #thrs.append(4)
            thrs = np.array(thrs)
            print(thrs)
            res = np.repeat(yh[np.newaxis,...], 4, axis=0) > np.repeat(thrs[np.newaxis,...], l, axis=0).T
            yt = res.sum(axis=0)
        else:
            yt = np.array([self.to_bins(y,self.rates) for y in yh])
            
        return yt
