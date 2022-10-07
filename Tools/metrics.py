from pycm import ConfusionMatrix
import numpy as np

class Metric():
    """Base class for all metrics. 
    """
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class ConfMat():
    """
        a wrapper of pycm's ConfusionMatrix
    """
    def __init__(self, size:int=3) -> None:
        self.size = size
        self.confmat = np.zeros((self.size, self.size), dtype=int)

    def __call__(self, pred, label) -> None:
       for p, t in zip(pred, label):
        self.confmat[t][p] += 1

    def getcm(self):
        cm_dict = {}
        for i in range(self.size):
            cm_dict[i] = {k:int(v) for k, v in zip(range(self.size), self.confmat[i])}  

        cm = ConfusionMatrix(matrix=cm_dict)
        return cm
        
    def reset(self):
        self.confmat = np.zeros((self.size, self.size))

        
