import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None
        self._input = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass


class Module_lineare(Module):

    def __init__(self, input_size, output_size , biais = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.biais = biais
        self._parameters = {
            'weights':np.random.random((input_size, output_size))-0.5,
            'biais': np.random.randn(output_size)-0.5,
        }
        self._gradient = dict()
        self.zero_grad()



    def zero_grad(self):
        ## Annule gradient
        self._gradient['weights'] = np.zeros((self.input_size, self.output_size))
        self._gradient['biais'] = np.zeros(self.output_size)

    def forward(self, X):
        ## Calcule la passe forward
        # X : (n , d_input) , W : (d_input , d_output)
        assert X.shape[1] == self.input_size
        self._input = X.copy()
        if self.biais :
            return X @ self._parameters['weights'] + self._parameters['biais']
        return X @ self._parameters['weights'] 
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size
        assert input.shape[0] == delta.shape[0]
        self._gradient['weights'] += input.T @ delta
        if self.biais :
            self._gradient['biais'] += np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
        # Calcul du gradient par rapport aux entrées
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size
        assert input.shape[0] == delta.shape[0]
        return delta @ self._parameters['weights'].T

    def update_parameters(self, gradient_step):
        # Mise à jour des paramètres avec un certain pas d'apprentissage
        self._parameters['weights'] -= gradient_step * self._gradient['weights']
        if self.biais :
            self._parameters['biais'] -= gradient_step * self._gradient['biais']

    def set_parameters(self, parameters):
        assert parameters['weights'].shape == self._parameters['weights'].shape
        assert parameters['biais'].shape == self._parameters['biais'].shape
        self._parameters['weights'] = parameters['weights'].copy()
        self._parameters['biais'] = parameters['biais'].copy()

    
    def get_parameters(self):
        parameters = dict()
        parameters['weights'] = self._parameters['weights'].copy()
        parameters['biais'] = self._parameters['biais'].copy()
        return parameters
    

#-------------------AJOUT DE LINEAIRE
class Softmax(Module):
    def __init__(self ):
        super().__init__()

    def forward(self, x):
        ## Calcule la passe forward
        self._input = x
        eps = 1e-4
        exps = np.exp(x) 
        return exps / (np.sum(exps, axis=1, keepdims=True) + eps)

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        eps = 1e-4
        exps = np.exp(input)
        q = exps / (np.sum(exps, axis=1, keepdims=True) + eps )
        return delta * q * (1 - q)