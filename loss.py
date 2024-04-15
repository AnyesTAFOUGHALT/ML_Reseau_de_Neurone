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

        #return np.sum((y - yhat)**2, axis=1)  / y.shape[0]
        return np.sum((y - yhat)**2, axis=1) 

    def backward(self, y, yhat):
        assert y.shape == yhat.shape
        return -2 * (y - yhat) 
