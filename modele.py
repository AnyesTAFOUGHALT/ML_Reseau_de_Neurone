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
    
#-------------------------- ReLu -----------------------#
    
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

    def update_parameters(self, gradient_step=1e-3):     
        pass

    def backward_update_gradient(self, input, delta):
        pass

#-------------------------- Conv1D -----------------------#   
class Conv1D(Module):
    def __init__(self,k_size,chan_in,chan_out,stride):
        """
        k_size : kernel size
        chan_in : number of input channels
        chan_out : number of output channels
        stride : stride value
        """
        self.k_size = k_size
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.stride = stride

        np.random.seed(0)
        self._parameters =2*(np.random.random((k_size,chan_in,chan_out))-0.5) * 1e-1 
        self._gradient = np.zeros((k_size,chan_in,chan_out))

    def forward(self,X):
        """
            X est de taille  (batch,length,chan_in)
            output est de taille (batch, (length-k_size)/stride +1,chan_out)
        """
        self._input = X.copy()
        batch_size,input_length,c = X.shape

        result = np.zeros((batch_size, (input_length-self.k_size)//self.stride + 1, self.chan_out))
        for n in range(batch_size):
            A = X[n]
            for f in range(self.chan_out):
                weights = self._parameters[:,:,f] 

                for i in range(0,result.shape[1]):
                    j = i*self.stride
                    B = A[j:j+self.k_size] 
                    
                    result[n,i,f] = (B * weights).sum()

        return result

    def backward_update_gradient(self, input, delta):
        """
        input de taille (batch, length, chan_in)
        delta de taille (batch, output_length, chan_out)
        """
        assert input.shape[2] == self.chan_in
        assert delta.shape[2] == self.chan_out
        assert delta.shape[1] == (input.shape[1]-self.k_size)//self.stride + 1
        assert delta.shape[0] == input.shape[0]
        batch_size, length_out, chan_out = delta.shape
        for n in range(batch_size):
            A = input[n] 
            for z in range(chan_out):
                for i in range(length_out):
                    Xs = A[i:i+self.k_size] 
                    delta0 = delta[n,i,z]
                    self._gradient[:,:,z] += Xs*delta0
    def update_parameters(self, gradient_step=1e-5):
        pass

    def backward_delta(self, input, delta):
        """
        input de taille (batch, length, chan_in)
        delta de taille (batch, output_length, chan_out)
        Returns:
        - delta de taille (batch, length, chan_in)
        """   

        b, length_out, chan_out = delta.shape
        
        result = np.zeros(input.shape)
        result = np.array(result, dtype=np.float64)

        for n in range(b):
            A = input[n] 
            for z in range(chan_out):
                weights = self._parameters[:,:,z] 
                for i in range(length_out):
                    delta0 = delta[n,i,z]
                    result[n,i:i+self.k_size,:] += weights*delta0
        return result
    
    def zero_grad(self):
        self._gradient = np.zeros(self._gradient.shape)

#-------------------------- MaxPool1D -----------------------#  
class MaxPool1D(Module):
    def __init__(self,k_size,stride):
        """
        k_size : kernel size
        stride : stride value
        """
        self.k_size = k_size
        self.stride = stride
        self.max_idx = None
    
    def forward(self,X):
        """
        data: input data of shape (batch, length, chan_in)
        Returns:
        - output of shape (batch, (length - k_size) // stride + 1, chan_in)
        """
        self._input = X.copy()
        batch_size,input_length,chan_in = X.shape

        result = np.zeros((batch_size, (input_length-self.k_size)//self.stride + 1, chan_in))
        self.max_idx = np.zeros(result.shape)

        for i in range(0, result.shape[1]):
            
            self.max_idx[:,i] = np.argmax(X[:, (i * self.stride): (i * self.stride + self.k_size)],
                                                 axis=1) + i * self.stride
            result[:,i,:] = np.max(X[:,(i*self.stride) : (i*self.stride + self.k_size)], axis=1)
        self.max_idx = self.max_idx.astype(int)
        return result

    def update_parameters(self, gradient_step=1e-5):      
        pass
        

    def backward_update_gradient(self, input, delta):      
        pass

    def backward_delta(self, input, delta):
        """
        input: input data of shape (batch, length, chan_in)
        delta: delta from the next layer of shape (batch, output_length, chan_in)
        Returns:
        - delta of shape (batch, length, chan_in)
        """
        batch_size,input_length,chan_in = input.shape        
        result = np.zeros_like(input)
        
        for n in range(batch_size):
            A = input[n]
            ind = self.max_idx[n]
            for i in range(ind.shape[0]):
                for j in range(ind.shape[1]):
                    result[n,ind[i,j],j] = delta[n,i,j]
        
        return result

#-------------------------- Flatten -----------------------#  
    
class Flatten(Module):
    def __init__(self):
        pass
    
    def forward(self,X):
        '''
        params:
        -------
        X : dim (batch,length,chan_in)

        return:
        -------
        dim (batch,length*chan_in)
        '''
        self._input = X.copy()
        return X.reshape((len(X),-1))
        
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        '''
        param:
        ------
        input : resPool, dim (batch,length,chan_in)
        delta : delta of lin, dim (batch,length*chan_out)
        
        return:
        dim (batch,length,chan_in)
        '''
        return delta.reshape(input.shape)

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def update_parameters(self, gradient_step=1e-5):
        '''
        no parameter
        '''        
        pass    