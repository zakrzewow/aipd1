import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from typing import Callable, List, Tuple
from pydub import AudioSegment
from typing import List, Tuple, Callable
from pydub.utils import make_chunks
from modules.frame import Frame


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