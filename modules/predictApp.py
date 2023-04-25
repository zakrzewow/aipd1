import numpy as np
import plotly.graph_objs as go
from scipy.signal import get_window
from modules.app import App


class PredictApp(App):

    def __init__(self, filepath_or_bytes, FFT_size, hop_size=15,  normalize=True):
        self.read_wav(filepath_or_bytes, normalize=normalize)
        self.FFT_size = FFT_size
        self.hop_size = hop_size
        self.filepath_or_bytes = filepath_or_bytes

    def frame_generator(self, FFT_size, hop_size=15):
        samples = np.pad(self.samples, int(FFT_size/2), mode = "reflect")
        frame_len = np.round(self.frame_rate * hop_size / 1000).astype(int)
        frame_num = int((len(self.samples - FFT_size) / frame_len)) + 1
        frames = np.zesros((frame_num, FFT_size))

        for n in range(frame_num):
            frames[n] = samples[n*frame_len:n*frame_len+FFT_size]

        return frames
    
    @staticmethod
    def rectangular_window(M):
        return np.ones(M)

    @staticmethod
    def triangular_window(M):
        return np.bartlett(M)
    
    def plot_window_comparision(self, window_name):

    

