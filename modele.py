import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

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
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self._parameters = {
            'weights': np.random.randn(input_size, output_size),
            'bias': np.zeros(output_size)
        }

    def zero_grad(self):
        ## Annule gradient
        self._gradient = None

    def forward(self, X):
        ## Calcule la passe forward
        # X : (n , d_input) , W : (d_input , d_output)
        return X @ self._parameters['weights'] + self._parameters['bias']
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        self.gradient['weights'] += np.dot(input.T, delta)
        self.gradient['bias'] += np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
        # Calcul du gradient par rapport aux entrées
        return np.dot(delta, self._parameters['weights'].T)

    def update_parameters(self, gradient_step):
        # Mise à jour des paramètres avec un certain pas d'apprentissage
        self._parameters['weights'] -= gradient_step * self.gradient['weights']
        self._parameters['bias'] -= gradient_step * self.gradient['bias']

