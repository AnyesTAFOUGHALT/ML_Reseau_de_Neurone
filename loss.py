import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class MSELoss(Loss):
    def forward(self, y, yhat):
        # y et yhat sont de taille batch * d
        assert y.shape == yhat.shape
        return np.sum((y - yhat)**2, axis=1) 

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return -2 * (y - yhat) 
    

class BCELoss(Loss):
    # Pour la classification binaire
    def forward(self, y, yhat):
        #params passé en entrée sont de la bonne taille
        assert y.shape == yhat.shape
        return - np.sum(y*np.log(yhat) + (1-y)*np.log(1-yhat) , axis=1 , keepdims=True)
    
    def backward(self, y, yhat):
        #params passé en entrée sont de la bonne taille
        assert y.shape == yhat.shape
        eps = 1e-4
        return - y/(yhat + eps) + (1-y) / (1-yhat + eps)
    
class CELoss(Loss) :
    # Pour la classification multi_classes
    def forward(self, y, yhat):
        # y et yhat sont de taille batch * d
        assert y.shape == yhat.shape
        exp_sum = np.sum(np.exp(yhat) , axis=1 ,  keepdims=True)
        return -np.sum(y * yhat , axis =1 ,  keepdims=True) + np.log(exp_sum)

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        eps = 1e-4
        exps = np.exp(yhat) 
        return -y + (exps / (np.sum(exps, axis=1, keepdims=True)+ eps))
    
