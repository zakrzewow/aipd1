from pydub import AudioSegment
import plotly.express as px
import numpy as np

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

        # stylowanie wykresów
        self._XAXIS_PARAMS = {
            "tickmode": "array",
            "tickvals": np.arange(0, len(self.samples), self.frame_rate / 2),
            "ticktext": [str(second).zfill(2) for second in np.arange(0, len(self.samples) / self.frame_rate, 0.5)],
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
    
    def plot_frame_level_feature(self, frame_level_func, frame_duration_miliseconds: int = 10):
        frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]
        x = [frame.timestamp for frame in frames]
        y = [frame_level_func(frame) for frame in frames]

        fig = px.line(x=x, y=y, template="plotly_white")
        fig.update_layout(
            xaxis=self._XAXIS_PARAMS,
            yaxis=self._YAXIS_PARAMS,
            yaxis_title="Value",
            xaxis_title="Time [s]",
            showlegend=False,
            hovermode=False,        
        )
        fig.update_traces(line_color="#16733e")
        return fig
    