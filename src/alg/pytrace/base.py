"""
    base module store some meta base definition 
    and the operator that bounded to trace itself:
        gather
        view
"""
import torch
import numpy as np
from torch import Tensor
from functools import reduce

class Tracer(Tensor):
    """
        simple wrapper of Tensor
    """
    def __init__(self, *args, **kwargs):
        super(Tracer, self).__init__()
        self._last_node = None
        self._is_leaf = False
        self._required = False
    
    def is_leaf(self):
        return self._is_leaf    
    
    def end(self):
        self._is_leaf = True
    
    def before(self):
        return self._last_node
    
    def store(self, node):
        assert isinstance(node, Operator)
        self._last_node = node
    
    def backward(self, show=False):
        if self._is_leaf:
            if show:
                print("BackProp:")
            self._last_node.backward(show=show)
        else:
            raise TypeError("Not end!!!")
    
    def requires_grad_(self):
        self._required = True
    
    def ask_grad(self):
        return self._required
    
    def __repr__(self):
        hints = super(Tracer, self).__repr__()
        hints = "base tracer" + hints[6:]
        return hints
    
    def base_Gather(self, index):
        gather_node = Gather_last(index)
        return gather_node(self)
    
    def base_View(self, *shape):
        view = View(*shape)
        return view(self)
    
    def base_Sum(self):
        summ = Sum()
        return summ(self)

class Operator:
    def __init__(self):
        self._name   : str      = None
        self._father : Operator = None
        self._son    : Operator = None
        self._out_x  : Operator = None
        self._tracing : bool   = True
        
    def __call__(self):
        raise NotImplementedError
    def _init_weight(self):
        for para in self.parameters():
            para.data.fill_(0.)
    def __repr__(self, **kwargs):
        if self._name is None:
            return "no name"
        else:
            return self._name
    def Pass(func):
        def ConnectAndStore(self : Operator, 
                            x, 
                            *args):
            if (not isinstance(x, Tracer)) or (x.before() is None):
                x = Input()(x)
            if self._tracing:
                self.connect(x.before())
            x = func(self, x, *args)
            if self._tracing:
                x.store(self)
                self._out_x = x
            return x
        return ConnectAndStore
    def End(func):
        def ConnectAndStore(self : Operator, 
                            x, 
                            *args):
            if (not isinstance(x, Tracer)) or (x.before() is None):
                x = Input()(x)
            if self._tracing:
                self.connect(x.before())
            x = func(self, x, *args)
            if self._tracing:
                x.store(self)
                self._out_x = x
            self._out_x.end()
            return x
        return ConnectAndStore
    def Back(func):
        def BackwardAndPass(self, grad=1., show=False):
            # print(self._father)
            # print(self)
            next_grad = func(self, grad=grad, show=show)
            self._father.backward(next_grad, show=show)
        return BackwardAndPass
    def backward(self, grad, show=False):
        raise NotImplementedError
    def getOutput(self):
        if not self._tracing:
            raise RuntimeError("When in eval mode, we will not store previous node")
        return self._out_x.float()
    def parameters(self):
        yield
    def branch(self,
               son):
        self._son = son 
    def connect(self,
                father):
        self._father = father
        father.branch(self)
    def eval(self):
        self._tracing = False
    def train(self):
        self._tracing = True

class Input(Operator):
    def __init__(self):
        super(Input, self).__init__()
        self._name = "Input"
    def __call__(self,x):
        if not isinstance(x, Tracer):
            x = Tracer(x).float()
        x.store(self)
        self._out_x = x
        return x
    
    def backward(self,grad, show=False):
        if show:
            print("↘︎", self.__repr__())
        if self._out_x.ask_grad():
            self._out_x.grad = grad

class Gather_last(Operator):
    def __init__(self, where):
        super(Gather_last, self).__init__()
        self._name = "Gather()"
        self.where = torch.Tensor(where.float()).long()
        self.index = torch.Tensor(np.arange(len(where))).long()
    
    @Operator.Pass
    def __call__(self, x):
        return Tracer(x[self.index, self.where].unsqueeze(-1))
    
    def backward(self, grad, show=False):
        if show:
            print("↘︎", self.__repr__())       
        before_x = self._father.getOutput()
        indice_grad = zeros_like(before_x)
        indice_grad[self.index, self.where] = grad.squeeze()
        self._father.backward(indice_grad, show=show)

class View(Operator):
    def __init__(self, *shape):
        super(View, self).__init__()
        self._name = f"View({shape})"
        self.to_shape = shape

    @Operator.Pass
    def __call__(self, x):
        x = x.reshape(self.to_shape)
        return Tracer(x)
        
    def backward(self, grad, show=False):
        if show:
            print("↘︎", self.__repr__())
        before_x = self._father.getOutput()
        self._father.backward(grad.reshape(before_x.shape), show=show)
    
class Sum(Operator):
    def __init__(self):
        super(Sum, self).__init__()
        self._name = "Sum()"
        
    @Operator.End
    def __call__(self, x):
        return Tracer(torch.sum(x).unsqueeze(-1))
    
    def backward(self, grad=1, show=False):
        if show:
            print("↘︎", self.__repr__())
        before_x = self._father.getOutput()
        sum_grad = ones_like(before_x)*grad
        self._father.backward(sum_grad, show=show)
        
        
def ones_like(tensor : Tracer):
    dtype = tensor.dtype
    shape = tensor.shape
    return torch.ones(shape, dtype=dtype)

def zeros_like(tensor : Tracer):
    dtype = tensor.dtype
    shape = tensor.shape
    return torch.zeros(shape, dtype=dtype)