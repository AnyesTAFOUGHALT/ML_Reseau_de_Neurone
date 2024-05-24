import numpy as np
from loss import *
import math
import copy


class Sequentiel:
    def __init__(self , modules):
        # for module in modules :
        #     assert isinstance(module, Module)
        self.modules = modules
        self.zero_grad()

    def add_module(self, module):
        # assert isinstance(module, Module)
        self.modules.append(module)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def forward(self, input):
        inputs = [input]
        for module in self.modules:
            inputs.append(module.forward(input))
            input = inputs[-1]
        
        return inputs

    def backward(self, delta):
        for module in reversed(self.modules) :
            module.backward_update_gradient(module._input , delta) 
            delta = module.backward_delta(module._input , delta)
        return delta

    def update_parameters(self, gradient_step):
        # Mise à jour des paramètres avec un certain pas d'apprentissage
        for module in self.modules :
            module.update_parameters(gradient_step)
            module.zero_grad()



class Optim:
    def __init__(self,net,loss,eps):
        self.net = net
        self.loss = loss
        self.eps = eps
    

    def step(self, batch_x, batch_y) :
        # pass forward
        y_pred = self.net.forward(batch_x)
        loss_value = self.loss.forward(batch_y, y_pred[-1])

        # pass backward
        gradient_loss = self.loss.backward(batch_y, y_pred[-1])
        delta = self.net.backward(gradient_loss )
        self.net.update_parameters(self.eps)

        #return y_pred , loss_value
        return loss_value.sum()

    def SGD(self , data , labels , batch_size , epochs ):
        assert len(data) == len(labels)
        N = len(data)
        index = np.arange(N)
        np.random.shuffle(index)

        loss_min = math.inf
        loses = []
        for i in range(epochs) :
            print("epoch ", i)
            loss_list = []
            for i in range(0 , N , batch_size) :
                batch_X = data[index[i:i+batch_size]]
                batch_Y = labels[index[i:i+batch_size]]
                loss_value = self.step(batch_X , batch_Y)
                loss_list.append(loss_value)
            mean_loss = np.mean(loss_list)
            if mean_loss < loss_min :
                loss_min = mean_loss
                best_network = copy.deepcopy(self.net)

            loses.append(mean_loss)
            
        self._net = best_network

        return loses

        

# def load_usps(fn):
#     with open(fn,"r") as f:
#         f.readline()
#         data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
#     tmp=np.array(data)
#     return tmp[:,1:],tmp[:,0].astype(int)

