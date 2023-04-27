import numpy as np
import plotly.graph_objs as go
from pydub import AudioSegment
from modules.window import Window
from modules.frame import Frame
import librosa
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import librosa
from scipy import fft
import numpy as np
import scipy.signal as signal
import scipy.fft as fft

class PredictApp():
    # stylowanie wykresów
    _XAXIS_PARAMS = {
        "tickmode": "array",
        "linecolor": "black",
        "gridcolor": "#c4cfc9",
        "showline": True,
        "mirror": True,
        "ticks": "outside",
        "title": "Time [s]"
    }
    _YAXIS_PARAMS = {
        "linecolor": "black",
        "gridcolor": "#c4cfc9",
        "showline": False,
        "ticks": "outside",
    }
    _DEFAULT_PARAMS = {
        "width": 900,
        "height": 400,
        "margin": dict(l=0, r=0, t=50, b=0),
        "hovermode": False,
        "template": "plotly_white",
        "showlegend": False,
    }

    def __init__(self, filepath_or_bytes, frame_duration_miliseconds: int = 5, normalize=True):
        self.read_wav(filepath_or_bytes, normalize=normalize)
        self.frame_duration_miliseconds = frame_duration_miliseconds
        self.frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]
        self.filepath_or_bytes = filepath_or_bytes

    def read_wav(self, filepath_or_bytes, normalize=True):
        self.audio_segment = AudioSegment.from_wav(filepath_or_bytes)
        self.frame_rate = self.audio_segment.frame_rate
        self.samples = np.asarray(self.audio_segment.get_array_of_samples(), dtype=float)
        if normalize:
            self.samples = self.samples / np.abs(self.samples).max()


    def frame_generator(self, frame_duration_miliseconds=10):
        n = int(self.frame_rate * frame_duration_miliseconds / 1000)
        offset = 0
        timestamp = 0.0
        duration = n / self.frame_rate
        for offset in range(0, len(self.samples), n):
            yield Frame(self.samples[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n
    
    def read_wav(self, filepath_or_bytes, normalize=True):
        self.audio_segment = AudioSegment.from_wav(filepath_or_bytes)
        self.frame_rate = self.audio_segment.frame_rate
        self.samples = np.asarray(self.audio_segment.get_array_of_samples(), dtype=float)
        if normalize:
            self.samples = self.samples / np.abs(self.samples).max()

    @staticmethod
    def frame_audio_ffr(samples, FFT_size = 2048, hop_size=15, sample_rate = 22050):
        samples = np.pad(samples, int(FFT_size/2), mode = "reflect")
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(samples) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num, FFT_size))

        for n in range(frame_num):
            frames[n] = samples[n*frame_len:n*frame_len+FFT_size]

        return frames

    @staticmethod
    def window_plot(audio_frammed_frr, audio_windowed, ind):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=audio_frammed_frr[ind], mode="lines", name="Original Frame"))
        fig.add_trace(go.Scatter(y=audio_windowed[ind], mode="lines", name="Frame After Windowing"))

        fig.update_layout(
            title="Window Plot",
            xaxis_title="Sample Index",
            yaxis_title="Amplitude",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig

    @staticmethod
    def freq_to_mel(freq):
        """
        Changes frequency to mel scale

        Parameters
        ----------
        freq  : frequency

        Returns
        ----------
        frequency in mel scale
        """  
        return 2595.0 * np.log10(1.0 + freq / 700.0)

    @staticmethod
    def mel_to_freq(mels):
        """
        Changes from mel scale to regular frequency

        Parameters
        ----------
        mels  : frequency in mels

        Returns
        ----------
        frequency in Hz scale
        """  
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)


    @staticmethod
    def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=22050):
        fmin_mel = PredictApp.freq_to_mel(fmin)
        fmax_mel = PredictApp.freq_to_mel(fmax)

        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
        freqs = PredictApp.mel_to_freq(mels)
        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

    @staticmethod
    def get_filters(filter_points, FFT_size):
        """
        Computes the MEL-spaced filterbank from filter points

        Parameters
        ----------
        filter_points : lineary spaced numpy array between the two MEL frequencies converted and nomalized to the FFT size (output of get_filter_points function)
        FFT_size      : int size of fast fourier transform (power of 2)

        Returns
        ----------
        MEL-spaced filterbank
        """
        filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
        
        for n in range(len(filter_points)-2):
            filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
            filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
        
        return filters

    @staticmethod 
    def plot_filters(filters):
        fig = go.Figure()

        for n in range(filters.shape[0]):
            fig.add_trace(go.Scatter(y=filters[n], mode="lines", name=f"Filter {n+1}"))

        fig.update_layout(
            title="Mel Filter Banks",
            xaxis_title="Frequency Bin",
            yaxis_title="Amplitude",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return fig


    @staticmethod
    def features_extractor(file_name):
        """
        Extracts MFCC features from audio file, calculated by librosa
        
        Parameters
        ----------
        file_name : path to the file
        
        Returns
        ----------
        mfccs_scaled_features : scaled matrix with cepstral coefficents - the product of perfoming MFCC
        """    
        #load the file (audio)
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        #we extract mfcc
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        #in order to find out scaled feature we do mean of transpose of value
        mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
        return mfccs_scaled_features

    @staticmethod
    def extract_features_from_directory(path="./samples/liczby/"):
        extracted_features = []
        label_mapping = {
            'zero': 0,
            'jeden': 1,
            'dwa': 2,
            'trzy': 3,
            'cztery': 4,
            'piec': 5,
            'szesc': 6,
            'siedem': 7,
            'osiem': 8,
            'dziewiec': 9,
            'dziesiec': 10
        }

        for file in tqdm(os.listdir(path)):
            if file.endswith(".wav"):
                file_path = os.path.join(path, file)
                file_label = os.path.splitext(file)[0].split('_')[0]  # Extract the label from the file name
                final_class_labels = label_mapping[file_label]  # Convert the string label to its corresponding number
                data = PredictApp.features_extractor(file_path)
                extracted_features.append([data, final_class_labels])

        extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
        return extracted_features_df


    






        
    

