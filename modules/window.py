from scipy.signal import get_window
import numpy as np

class Window:
    
    @staticmethod
    def rectangular_window(M):
        return np.ones(M)

    @staticmethod
    def triangular_window(M):
        return np.bartlett(M)

    @staticmethod
    def get_window(name, length):
        if name == "okno prostokątne":
            return Window.rectangular_window(length)
        elif name == "okno trójkątne":
            return Window.triangular_window(length)
        else:
            return  get_window(name, length)