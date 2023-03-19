from pydub import AudioSegment
import plotly.express as px
import numpy as np
from IPython.display import Audio
import IPython
import ipywidgets as widgets
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import IPython.display as ipd
from IPython.display import display
from IPython.display import clear_output
from IPython.core.interactiveshell import InteractiveShell
from pydub.effects import normalize
import librosa
InteractiveShell.ast_node_interactivity = "all"



class Frame:
    def __init__(self, samples, timestamp, duration):
        self.samples = samples
        self.timestamp = timestamp
        self.duration = duration

class App:

    def __init__(self, filepath: str, frame_duration_miliseconds: int = 10):
        self.read_wav(filepath)
        self.frame_duration_miliseconds = frame_duration_miliseconds
        self.frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]
        self.filepath = filepath

        # stylowanie wykresów
        self._XAXIS_PARAMS = {
            "tickmode": "array",
            # "tickvals": np.arange(0, len(self.samples), self.frame_rate / 2),
            # "ticktext": [str(second).zfill(2) for second in np.arange(0, len(self.samples) / self.frame_rate, 0.5)],
            "linecolor": "black",
            "gridcolor": "#c4cfc9",
            "showline": True,
            "mirror": True,
            "ticks": "outside",
        }
        self._YAXIS_PARAMS = {
            "linecolor": "black",
            "gridcolor": "#c4cfc9",
            "mirror": True,
            "ticks": "outside",
        }

    def read_wav(self, filepath: str):
        self.audio_segment = AudioSegment.from_wav(filepath)
        # Normalize the audio segment to -20 dBFS
        # normalized_segment = normalize(self.audio_segment).get_array_of_samples()

        # # Get the samples as an array of signed 16-bit integers
        # samples = normalized_segment
        self.samples = self.audio_segment.get_array_of_samples()
        self.frame_rate = self.audio_segment.frame_rate

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
        return np.sqrt(np.power(np.asarray(frame.samples, dtype="int64"), 2).mean())
    
    @staticmethod
    def STE(frame: Frame):
        return np.power(np.asarray(frame.samples, dtype="int64"), 2).mean()
    
    @staticmethod
    def ZCR(frame: Frame):
        count = 0
        for i in range(len(frame.samples)-1):
            if (frame.samples[i] >= 0 and frame.samples[i+1] < 0) or (frame.samples[i] < 0 and frame.samples[i+1] >= 0):
                count += 1
        zcr = count / len(frame.samples)
        return zcr
    
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
            xaxis=self._XAXIS_PARAMS,
            yaxis=self._YAXIS_PARAMS,
            yaxis_title="Amplitude",
            xaxis_title="Time [s]",
            showlegend=False,
            hovermode=False,        
        )
        fig.update_traces(line_color="#16733e")
        return fig
        
    def plot_frame_level_feature(self, frame_level_func, frame_level_func_name,
                                frame_duration_miliseconds: int = 10,
                                min_val: float = None, max_val: float = None):
        
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
            xaxis_title="Time [s]",
            hovermode=False,
            template="plotly_white",
            title = frame_level_func_name
        )

        return fig
    

    
    def get_frames_between_values(self, frame_level_func, min_val, max_val):
        return [frame for frame in self.frames if min_val <= frame_level_func(frame) <= max_val]
    

    def display_frames(self, frames):
        samples_to_display = np.concatenate([frame.samples for frame in frames])
        audio = Audio(data=samples_to_display, rate=self.frame_rate)
        return audio.data

    def display_frames_between_values(self, frame_level_func, min_val, max_val):
        frames_to_display = self.get_frames_between_values(frame_level_func, min_val, max_val)
        audio_data = self.display_frames(frames_to_display)
        return audio_data
        
    
    # def display_frames_between_values(self, frame_level_func, min_val=0,
    #                                   max_val=10**9, step=10**4):
    #     def update_display(min_max_val):
    #         min_val, max_val = min_max_val
    #         frames_to_display = self.get_frames_between_values(frame_level_func, min_val, max_val)
    #         if len(frames_to_display) == 0:
    #             print("Za mały zakres")
    #         else:
    #             self.display_frames(frames_to_display)

    #     # slider_label = widgets.Label(value='Range:', layout=widgets.Layout(width='200px'))
    #     slider = widgets.FloatRangeSlider(value=[min_val, max_val], min=0, max=max_val, step=step,
    #                                       description='', readout_format='.2f',
    #                                     layout=widgets.Layout(width='80%'),
    #                                     readout_style='width:50%;')



    #     widgets.interact(update_display, min_max_val=slider)*,

    def plot_frame_level_feature_analyse(self, frame_level_func,
                                          frame_duration_miliseconds=10,
                                            min_val=None, max_val=None):
        frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]
        x = [frame.timestamp for frame in frames]
        y = [frame_level_func(frame) for frame in frames]

        fig = go.Figure()

        # Add line trace
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color="#16733e"), showlegend=False))

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
            xaxis_title="Time [s]",
            hovermode=False,
            template="plotly_white"
        )

        # Create input widgets for min_val and max_val
        min_val_input = widgets.FloatText(value=min_val, description="Min Value")
        max_val_input = widgets.FloatText(value=max_val, description="Max Value")

        # Create a function to update the plot when the input fields are changed
        def update_plot(change):
            new_min_val = float(min_val_input.value) if min_val_input.value != "" else None
            new_max_val = float(max_val_input.value) if max_val_input.value != "" else None
            with out:
                clear_output()
                display(self.plot_frame_level_feature(frame_level_func, frame_duration_miliseconds, new_min_val, new_max_val))

        # Attach the update function to the change events of the input widgets
        min_val_input.observe(update_plot, names='value')
        max_val_input.observe(update_plot, names='value')

        # Create an output widget to display the plot
        out = widgets.Output()
        display(min_val_input, max_val_input, out)

