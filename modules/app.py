from pydub import AudioSegment
import plotly.express as px
import numpy as np

class App:
    def __init__(self, filepath: str):
        self.read_wav(filepath)

    def read_wav(self, filepath: str):
        self.audio_segment = AudioSegment.from_wav(filepath)
        self.samples = self.audio_segment.get_array_of_samples()
        self.frame_rate = self.audio_segment.frame_rate

    def plot_sample(self):
        fig = px.line(self.samples, template="plotly_white")
        fig.update_layout(
            xaxis={
                "tickmode": "array",
                "tickvals": np.arange(0, len(self.samples), self.frame_rate / 2),
                "ticktext": [str(second).zfill(2) for second in np.arange(0, len(self.samples) / self.frame_rate, 0.5)],
                "linecolor": "black",
                "gridcolor": "#c4cfc9",
                "showline": True,
                "mirror": True,
                "ticks": "outside",
            },
            yaxis={
                "linecolor": "black",
                "gridcolor": "#c4cfc9",
                "mirror": True,
                "ticks": "outside",
            },
            yaxis_title="Amplitude",
            xaxis_title="Time [s]",
            showlegend=False,
            hovermode=False,        
        )
        fig.update_traces(line_color="#16733e")
        return fig
    