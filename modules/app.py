import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from typing import Callable, List
from pydub import AudioSegment


class Frame:
    def __init__(self, samples, timestamp, duration):
        self.samples = samples
        self.timestamp = timestamp
        self.duration = duration

class App:
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
        "mirror": True,
        "ticks": "outside",
    }
    _DEFAULT_PARAMS = {
        "width": 800,
        "height": 400,
        "margin": dict(l=0, r=0, t=50, b=0),
        "hovermode": False,
        "template": "plotly_white",
        "showlegend": False,
    }

    def __init__(self, filepath: str, frame_duration_miliseconds: int = 10, normalize=True):
        self.read_wav(filepath, normalize=normalize)
        self.frame_duration_miliseconds = frame_duration_miliseconds
        self.frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]
        self.filepath = filepath

    def read_wav(self, filepath: str, normalize=True):
        self.audio_segment = AudioSegment.from_wav(filepath)
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

    @staticmethod
    def volume(frame: Frame):
        return np.sqrt(np.power(frame.samples, 2).mean())
    
    @staticmethod
    def STE(frame: Frame):
        return np.power(frame.samples, 2).mean()
    
    @staticmethod
    def ZCR(frame: Frame):
        return 2 / len(frame.samples) * np.sum(np.abs(np.sign(frame.samples[:-1]) - np.sign(frame.samples[1:])))
    
    @staticmethod
    def F0_Cor(frame: Frame, lag=5):
        s = 0
        for i in range(len(frame.samples) - lag):
            s = s + frame.samples[i] * frame.samples[i + lag]
        return s

    @staticmethod
    def F0_AMDF(frame: Frame, lag=5):
        s = 0
        for i in range(len(frame.samples) - lag):
            s = s + abs(frame.samples[i] - frame.samples[i + lag])
        return s
    
    @staticmethod
    def SR(frame: Frame, threshold=-60):
        ## TODO with volume  ZCR // 2 thresholds

        samples = np.asarray(frame.samples, dtype="int64")
        dbfs = librosa.amplitude_to_db(samples, ref=np.max)
        num_silent_samples = np.sum(dbfs < threshold)
        sr = num_silent_samples / len(frame.samples)
        return sr

    @staticmethod
    def LSTER(frame: Frame):
        """
        Computes the Long-Term Spectral Energy Ratio (LSTER) of a frame.
        """
        # Compute the STFT of the frame
        window_size = len(frame.samples)
        n_fft = 2048
        hop_length = n_fft // 4
        stft = np.abs(
            librosa.stft(frame.samples, n_fft=n_fft, hop_length=hop_length, win_length=window_size)
        )
        
        # Compute the energy in each frequency band
        num_bands = 4
        band_energy = np.zeros(num_bands)
        band_freqs = np.linspace(0, self.frame_rate / 2, stft.shape[0])
        freq_bounds = np.linspace(0, self.frame_rate / 2, num_bands + 1)
        for i in range(num_bands):
            freq_idx = np.where((band_freqs >= freq_bounds[i]) & (band_freqs < freq_bounds[i+1]))[0]
            band_energy[i] = np.sum(stft[freq_idx, :])
        
        # Compute the LSTER value
        lster = np.sum(band_energy[1:]) / np.sum(band_energy)
        return lster
    
    #staticmethod
    def VSTD(self, frames):
        Volumes = [self.volume(item) for item in frames]
        return np.std(Volumes)/max(Volumes)

    #staticmethod
    def VDR(self, frames):
        Volumes = [self.volume(item) for item in frames]
        return (max(Volumes)-min(Volumes))/max(Volumes)


    @staticmethod
    def ZSTD(self, frames):
        zcr = [self.ZCR(item) for item in frames]
        return np.std(zcr)

    @staticmethod
    def HZCRR(self, samples, frames):
        avgZCR = sum([self.ZCR(item) for item in frames])/len(frames)
        s = 0
        for i in range(len(frames)):
            s = s + np.sign(0.5*avgZCR-self.ZCR(frames[i]))+1
        hzcrr = 1/(2*len(frames))*s
        return hzcrr
       
    def plot_sample(self):
        fig = px.line(self.samples, template="plotly_white")
        fig.update_layout(
            xaxis={
                **self._XAXIS_PARAMS,
                "tickvals": np.arange(0, len(self.samples), self.frame_rate / 2),
                "ticktext": [str(second).zfill(2) for second in np.arange(0, len(self.samples) / self.frame_rate, 0.5)],
            },
            yaxis=self._YAXIS_PARAMS,
            yaxis_title="Amplitude",
            **self._DEFAULT_PARAMS      
        )
        fig.update_traces(line_color="#16733e")
        return fig
        
    def plot_frame_level_feature(
            self, 
            frame_level_func: Callable[[Frame], float], 
            plot_title: str,
            frame_duration_miliseconds: int = 10,
            min_val: float = None, 
            max_val: float = None
        ):
        
        frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]
        x = [frame.timestamp for frame in frames]
        y = [frame_level_func(frame) for frame in frames]

        fig = go.Figure()

        # Add line trace
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', line=dict(color="#16733e"),
                        showlegend=False)
        )

        # Color regions between min_val and max_val
        if min_val is not None and max_val is not None:
            for i in range(len(y)-1):
                if min_val <= y[i] <= max_val or min_val <= y[i+1] <= max_val:
                    fig.add_shape(
                        type="rect", x0=x[i], y0=min_val, x1=x[i+1], y1=max_val,
                        fillcolor="#e63946", opacity=0.2
                    )

        fig.update_layout(
            xaxis=self._XAXIS_PARAMS,
            yaxis=self._YAXIS_PARAMS,
            yaxis_title="Value",
            title=plot_title,
            title_x=0.1,
            title_y=0.95,
            **self._DEFAULT_PARAMS
        )

        return fig
    
    def get_frame_level_export_data(self, frame_duration_miliseconds):
        records = {}

        frame_funcs = {
            "Volume": self.volume,
            "Short-Time Energy": self.STE,
            "Zero Crossing Rate": self.ZCR,
            "Silent ratio": self.SR
        }

        frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]

        for name, func in frame_funcs.items():
            records[name] = [func(frame) for frame in frames]
        return pd.DataFrame(records).round(4)
    
    def get_frame_level_func_range(self, frame_level_func, frame_duration_miliseconds):
        frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]
        values = [frame_level_func(frame) for frame in frames]
        return frames, values, float(max(values))

    def display_frames_signal_between_values(self, frames, func_values, min_val, max_val):
        frames_to_display = [frame for idx, frame in enumerate(frames) if min_val <= func_values[idx] <= max_val]
        samples = self.__frames_to_samples(frames_to_display)
        return samples
    
    def __frames_to_samples(self, frames: List[Frame]) -> np.array:
        return np.concatenate([frame.samples for frame in frames])
