import numpy as np
import modele
import utils as ut

class TanH(modele.Module):

    def __init__(self):
        super().__init__()
        self._parameters = None
        self._gradient = None


    def forward(self, X):
        ## Calcule la passe forward
        return np.tanh(X) 


    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # Calcul du gradient par rapport aux entrées
        return (1-np.tanh(input)**2) * delta 
    

class Sigmoide(modele.Module):

    def __init__(self):
        self._parameters = None
        self._gradient = None


    def forward(self, X):
        ## Calcule la passe forward
        return 1/(1+np.exp(-X))

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # Calcul du gradient par rapport aux entrées
        return delta * ut.sigmoid_derivative(input)
