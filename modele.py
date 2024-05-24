import numpy as np
import utils as  ut 

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


#------------------- Linéaire -------------------#

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
    

#------------------- Non_Linéaire -------------------#

class TanH(Module):

    def __init__(self):
        super().__init__()
        self._parameters = None
        self._gradient = None


    def forward(self, X):
        ## Calcule la passe forward
        self._input = X
        return np.tanh(X) 


    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # Calcul du gradient par rapport aux entrées
        return (1-np.tanh(input)**2) * delta 
    

class Sigmoide(Module):

    def __init__(self):
        self._parameters = None
        self._gradient = None


    def forward(self, X):
        ## Calcule la passe forward
        self._input = X
        return 1/(1+np.exp(-X))

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # Calcul du gradient par rapport aux entrées
        return delta * ut.sigmoid_derivative(input)


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
    
#-------------------------- Convolution -----------------------#
    
class ReLu(Module):
    def __init__(self, threshold=0 ):
        super().__init__()
        self._threshold = threshold

    def forward(self, x):
        ## Calcule la passe forward
        self._input = x
        return np.where(x>self._threshold,x,0.)
    
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        return np.where(input>self._threshold,1.,0.) * delta 

#-------------------------- Conv1D -----------------------#   
class Conv1D(Module):
    def __init__(self, k_size, chan_in, chan_out, stride=1):
        """
        k_size : kernel size
        chan_in : number of input channels
        chan_out : number of output channels
        stride : stride value
        """
        super().__init__()
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride
        self._parameters = np.random.randn(k_size, chan_in, chan_out)
        self._gradient = np.zeros((k_size, chan_in, chan_out))

    def forward(self, data):
        """
            X est de taille  (batch,length,chan_in)
            output est de taille (batch, (length-k_size)/stride +1,chan_out)
        """
        batch_size, input_length, _ = data.shape
        output_length = (input_length - self.k_size) // self.stride + 1
        output = np.zeros((batch_size, output_length, self.chan_out))
        for i in range(output_length):
            output[:, i, :] = np.sum(data[:, i:i+self.k_size, :, np.newaxis] * self._parameters, axis=(1, 2))
        return output

    def zero_grad(self):
        self._gradient = np.zeros_like(self._parameters)

    def backward_update_gradient(self, input, delta):
        """
        input de taille (batch, length, chan_in)
        delta de taille (batch, output_length, chan_out)
        """
        batch_size, input_length, _ = input.shape
        output_length, _ = delta.shape[1:]
        for b in range(batch_size):
            for i in range(output_length):
                input_segment = input[b, i:i+self.k_size, :]
                self._gradient += np.outer(input_segment.flatten(), delta[b, i, :])

    def backward_delta(self, input, delta):
        """
        input de taille (batch, length, chan_in)
        delta de taille (batch, output_length, chan_out)
        Returns:
        - delta de taille (batch, length, chan_in)
        """
        batch_size, input_length, _ = input.shape
        output_length, _ = delta.shape[1:]
        out = np.zeros_like(input)
        for b in range(batch_size):
            for i in range(output_length):
                out[b, i:i+self.k_size, :] += np.matmul(delta[b, i, :], self._parameters.reshape(-1, self.chan_out).T).reshape(self.k_size, self.chan_in)
        return out

    def update_parameters(self, gradient_step):
        self._parameters -= gradient_step * self._gradient


#-------------------------- MaxPool1D -----------------------#  
class MaxPool1D(Module):
    def __init__(self, k_size, stride):
        """
        k_size : kernel size
        stride : stride value
        """
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, data):
        """
        data: input data of shape (batch, length, chan_in)
        Returns:
        - output of shape (batch, (length - k_size) // stride + 1, chan_in)
        """
        batch_size, input_length, chan_in = data.shape
        output_length = (input_length - self.k_size) // self.stride + 1
        output = np.zeros((batch_size, output_length, chan_in))
        for i in range(output_length):
            output[:, i, :] = np.max(data[:, i:i+self.k_size, :], axis=1)
        return output

    def backward_delta(self, input, delta):
        """
        input: input data of shape (batch, length, chan_in)
        delta: delta from the next layer of shape (batch, output_length, chan_in)
        Returns:
        - delta of shape (batch, length, chan_in)
        """
        batch_size, input_length, chan_in = input.shape
        output_length = delta.shape[1]
        out = np.zeros_like(input)
        for b in range(batch_size):
            for i in range(output_length):
                indices = np.argmax(input[b, i*self.stride:i*self.stride+self.k_size, :], axis=0)
                out[b, i*self.stride:i*self.stride+self.k_size, :] = delta[b, i, :] * (indices[:, np.newaxis] == np.arange(chan_in))
        return out


#-------------------------- Flatten -----------------------#  
class Flatten(Module):

    def __init__(self ):
        super().__init__()

    def forward(self, data):
        """
        data: input data of shape (batch, length, chan_in)
        Returns:
        - flattened output of shape (batch, length * chan_in)
        """
        batch_size, length, chan_in = data.shape
        return data.reshape(batch_size, -1)

    def backward_delta(self, input, delta):
        """
        input: input data of shape (batch, length, chan_in)
        delta: delta from the next layer of shape (batch, length * chan_in)
        Returns:
        - delta of shape (batch, length, chan_in)
        """
        return delta.reshape(input.shape)






