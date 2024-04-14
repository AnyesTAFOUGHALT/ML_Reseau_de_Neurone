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
        assert X.shape[1] == self.input_size
        return np.tanh(X) 


    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # Calcul du gradient par rapport aux entrées
        assert input.shape[1] == self.input_size # chaque exemple x doit avoir une dimension egale au nombre d'entrees supportées par le reseau de neurones 
        assert delta.shape[1] == self.output_size # pour chaque delta on a une valeur pour calcule pour chaque noeurone de sortie
        assert input.shape[0] == delta.shape[0] # autant de d'exmeples que de delta genere
        return (1-np.tanh(input)**2) * delta 
    

class Sigmoide(modele.Module):

    def __init__(self):
        self._parameters = None
        self._gradient = None


    def forward(self, X):
        ## Calcule la passe forward
        assert X.shape[1] == self.input_size
        return 1/(1+np.exp(-X))

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # Calcul du gradient par rapport aux entrées
        assert input.shape[1] == self.input_size # chaque exemple x doit avoir une dimension egale au nombre d'entrees supportées par le reseau de neurones 
        assert delta.shape[1] == self.output_size # pour chaque delta on a une valeur pour calcule pour chaque noeurone de sortie
        assert input.shape[0] == delta.shape[0] # autant de d'exmeples que de delta genere
        return delta * ut.sigmoid_derivative(input).T
