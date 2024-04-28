from modele import Module
from loss  import Loss
import numpy as np
import math
import copy

class Sequentiel:
    def __init__(self , modules):
        # for module in modules :
        #     assert isinstance(module, Module)
        self._modules = modules
        self.zero_grad()

    def add_module(self, module):
        assert isinstance(module, Module)
        self._modules.append(module)

    def forward(self, X):
        input = X.copy()
        for module in self._modules :
            output = module.forward(input)
            input = output.copy()
        return output
    
    def zero_grad(self):
        for module in self._modules:
            module.zero_grad()

    def backward(self, delta):
        for module in reversed(self._modules) :
            module.backward_update_gradient(module._input , delta) 
            delta = module.backward_delta(module._input , delta)
        return delta

    def update_parameters(self, gradient_step):
        # Mise à jour des paramètres avec un certain pas d'apprentissage
        for module in self._modules :
            if module._gradient != None :
                module.update_parameters(gradient_step)
                module.zero_grad()

class Optim:
    def __init__(self , net, loss, eps) :
        # assert isinstance(net, Sequentiel)
        # assert isinstance(loss, Loss)
        self._net = net
        self._loss = loss
        self._eps = eps

    def step(self, batch_x, batch_y) :
        # pass forward
        y_pred = self._net.forward(batch_x)
        loss_value = self._loss.forward(batch_y, y_pred)

        # pass backward
        gradient_loss = self._loss.backward(batch_y, y_pred)
        delta = self._net.backward(gradient_loss)

        # Mise à jour des paramètres
        self._net.update_parameters(self._eps)

        return y_pred , loss_value
    
    def SGD(self , data , labels , batch_size , epochs ) :
        assert len(data) == len(labels)
        N = len(data)
        index = np.arange(N)
        np.random.shuffle(index)

        loss_min = math.inf
        loss_list = []
        best_network = None

        for _ in range(epochs) :
            for i in range(0 , N , batch_size) :
                batch_X = data[index[i:i+batch_size]]
                batch_Y = labels[index[i:i+batch_size]]
                y_pred , loss_value = self.step(batch_X , batch_Y)
                loss_list.append(np.mean(loss_value))
                if np.mean(loss_value) < loss_min :
                    loss_min = np.mean(loss_value)
                    best_network = copy.deepcopy(self._net)
        self._net = best_network
    
    # def accuracy(self , data , labels) :
    #     pred = np.where(self._net.forward(data) >= 0.5,1, 0)
    #     return np.mean(np.where(labels == pred, 1 , 0 ))