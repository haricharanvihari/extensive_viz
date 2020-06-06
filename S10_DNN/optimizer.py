import torch.optim as optim

def OptimizerFactory(o_type ="SGD"): 
  
    """Factory Method"""
    optimizations = { 
        "SGD": SGDOptimizer, 
        "Adam": AdamOptimizer,
    } 
  
    return optimizations[o_type]()

class SGDOptimizer(object):
  def __init__(self):
    super(SGDOptimizer, self).__init__()

  def load(self, **kwargs):
    self.model_params = kwargs.get("params", None)
    self.lr = kwargs.get("lr", 0.001)
    self.momentum = kwargs.get("momentum", 0)
    self.dampening = kwargs.get("dampening", 0)
    self.weight_decay = kwargs.get("weight_decay", 0)
    self.nesterov = kwargs.get("nesterov", False)
    return optim.SGD(self.model_params, lr=self.lr, momentum=self.momentum, 
                     weight_decay=self.weight_decay, dampening=self.dampening, nesterov=self.nesterov)

class AdamOptimizer(object):
  def __init__(self):
    super(AdamOptimizer, self).__init__()

  def load(self, **kwargs):
    self.model_params = kwargs.get("params", None)
    self.lr = kwargs.get("lr", 0.001)
    self.betas = kwargs.get("betas", (0.9, 0.999))
    self.eps = kwargs.get("eps", 1e-08)
    self.weight_decay = kwargs.get("weight_decay", 0)
    self.amsgrad = kwargs.get("amsgrad", False)
    return optim.Adam(self.model_params, lr=self.lr, weight_decay=self.weight_decay)