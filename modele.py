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
    def __init__(self, input_size, output_size , biais = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.biais = biais
        self._parameters = {
            'weights': np.random.randn(input_size, output_size),
            'biais': np.zeros(output_size)
        }

        self.zero_grad()

    def zero_grad(self):
        ## Annule gradient
        self._gradient['weights'] = np.zeros((self.input_size, self.output_size))
        self._gradient['biais'] = np.zeros(self.output_size)

    def forward(self, X):
        ## Calcule la passe forward
        # X : (n , d_input) , W : (d_input , d_output)
        assert X.shape[1] == self.input_size
        if self.biais :
            return X @ self._parameters['weights'] + self._parameters['biais']
        return X @ self._parameters['weights'] 
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size
        assert input.shape[0] == delta.shape[0]
        self.gradient['weights'] += input.T @ delta / len(input)
        if self.biais :
            self.gradient['biais'] += np.mean(delta, axis=0)

    def backward_delta(self, input, delta):
        # Calcul du gradient par rapport aux entrées
        assert input.shape[1] == self.input_size
        assert delta.shape[1] == self.output_size
        assert input.shape[0] == delta.shape[0]
        return delta @ self._parameters['weights'].T

    def update_parameters(self, gradient_step):
        # Mise à jour des paramètres avec un certain pas d'apprentissage
        self._parameters['weights'] -= gradient_step * self.gradient['weights']
        if self.biais :
            self._parameters['biais'] -= gradient_step * self.gradient['biais']

