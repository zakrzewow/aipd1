import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from typing import Callable, List, Tuple
from pydub import AudioSegment
from typing import List, Tuple, Callable
from pydub.utils import make_chunks
import math



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


    @staticmethod
    def volume(frame: Frame):
        return np.sqrt(np.power(frame.samples, 2).mean())
    

    @staticmethod
    def compute_peak_values(frame: Frame, window_size: int = 10) -> float:
        """
        Compute the peak values of an audio frame in a given time window.

        Parameters:
        frame (Frame): The audio frame as a `Frame` object.
        window_size (int): The size of the time window in samples.

        Returns:
        float: The peak value of the signal in the time window.
        """

        # Compute the maximum peak value over all windows
        return np.max([np.max(np.abs(frame.samples[i:i+window_size])) for i in range(0, len(frame.samples)-window_size, window_size)])


    @staticmethod
    def RMS(frame: Frame) -> float:
        """
        Function to calculate the Root Mean Square (RMS) of a frame.

        Parameters:
        frame (Frame): A Frame object.

        Returns:
        float: The RMS value for the frame.
        """
        signal = frame.samples
        rms = np.sqrt(np.mean(signal ** 2))
        return rms
    
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
    def SR(frame: Frame, vol_min_value=0.0, vol_max_value=0.05, zcr_min_value=0.5, zcr_max_value=2):
        volume = App.volume(frame)
        zcr = App.ZCR(frame)
        return vol_min_value <= volume <= vol_max_value and zcr_min_value <= zcr <= zcr_max_value

    @staticmethod
    def autocorrelation(frame: Frame) -> np.ndarray:
        """
        Function to calculate the normalized autocorrelation of a frame.

        Parameters:
        frame (Frame): A Frame object.

        Returns:
        np.ndarray: The normalized autocorrelation values for the frame.
        """
        signal = frame.samples
        n = len(signal)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[n - 1:]
        autocorr /= autocorr.max()  # Normalize the autocorrelation values
        
        # Calculate sample rate
        sample_rate = len(signal) / frame.duration
        
        return autocorr, sample_rate
    

    def chunk_frame_generator(self, chunk: AudioSegment, frame_duration_miliseconds=10):
        frame_rate = chunk.frame_rate
        samples = np.asarray(chunk.get_array_of_samples(), dtype=float)
        samples = samples / np.abs(samples).max()
        n = int(frame_rate * frame_duration_miliseconds / 1000)
        offset = 0
        timestamp = 0.0
        duration = n / frame_rate
        for offset in range(0, len(samples), n):
            yield Frame(samples[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n


    @staticmethod
    def LSTER(frames: List[Frame], window_duration: float = 1) -> float:
        """
        Function to calculate the Low Short Time Energy Ratio (LSTER).
        Parameters:
        frames (List[Frame]): A list of Frame objects.
        window_duration (float): The duration of the window to compute the average STE. Default is 1.0 second.
        Returns:
        float: The Low Short Time Energy Ratio (LSTER).
        """
        total_frames = len(frames)
        frame_duration = frames[0].duration
        window_size = int(window_duration / frame_duration)

        ste_values = [App.STE(frame) for frame in frames]
        low_ste_count = 0

        for i in range(total_frames - window_size):
            window_ste_avg = np.mean(ste_values[i:i+window_size])
            if ste_values[i] < 0.5 * window_ste_avg:
                low_ste_count += 1

        lster = low_ste_count / total_frames
        return lster
    

    def proper_LSTER(self):
        chunk_length_ms = 1000
        chunks = make_chunks(self.audio_segment, chunk_length_ms)
        
        frame_duration_miliseconds = 25

        lster_list = []
        for index, chunk in enumerate(chunks):
            frames = [frame for frame in self.chunk_frame_generator(chunk, frame_duration_miliseconds)]

            print(f"Chunk {index + 1}/{len(chunks)}: {len(frames)} frames")

            if len(frames) > 0:
                lster_value = App.LSTER(frames)
                print(f"LSTER value for chunk {index + 1}: {lster_value}")
                lster_list.append(lster_value)

        return lster_list

    def plot_proper_music_metric(self, lster_list: List[float], plot_title: str):
        x = list(range(len(lster_list)))
        y = lster_list

        fig = go.Figure()

        # Add line trace
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines', line=dict(color="#16733e"),
                        showlegend=False)
        )

        fig.update_layout(
            xaxis_title="Chunk index",
            yaxis_title=plot_title,
            title=plot_title,
            title_x=0.05,
            title_y=0.95,
        )

        return fig
    

    @staticmethod
    def energy_entropy(frames, K=10):
        """
        Calculates the energy entropy of a list of audio frames.

        Parameters:
        frames (List[Frame]): A list of Frame objects.
        K (int): The number of samples in each segment. Default is 10.

        Returns:
        float: The energy entropy of the given frames.
        """

        energy_values = []

        for frame in frames:
            total_samples = len(frame.samples)
            num_segments = total_samples // K
            segment_energies = []

            for i in range(num_segments):
                segment = frame.samples[i * K:(i + 1) * K]
                segment_energy = sum(x*x for x in segment)
                segment_energies.append(segment_energy)

            total_energy = sum(segment_energies)
            segment_probs = [x/total_energy for x in segment_energies]
            segment_entropy = -sum(p*math.log2(p) for p in segment_probs)
            energy_values.append(segment_entropy)

        energy_entropy = sum(energy_values) / len(energy_values)
        return energy_entropy
    
    @staticmethod
    def VSTD(frames):
        Volumes = [App.volume(item) for item in frames]
        return np.std(Volumes)/max(Volumes)

    @staticmethod
    def VDR(frames):
        Volumes = [App.volume(item) for item in frames]
        return (max(Volumes)-min(Volumes))/max(Volumes)
    
    @staticmethod
    def VU(frames):
        Volumes = [App.volume(item) for item in frames]
        VSTD = np.std(Volumes)
        VMA = np.mean(Volumes)
        return VSTD/VMA

    @staticmethod
    def ZSTD(frames):
        zcr = [App.ZCR(item) for item in frames]
        return np.std(zcr)

    @staticmethod
    def HZCRR(frames):
        N = len(frames)
        zcr_values = [App.ZCR(frame) for frame in frames]
        avZCR = np.mean(zcr_values)
        
        HZCRR = 0
        for i in range(N):
            HZCRR += (np.sign(zcr_values[i] - 1.5 * avZCR) + 1) / 2
        HZCRR /= N

        return HZCRR
    

    @staticmethod
    def spectral_centroid(frame: Frame) -> float:
        """
        Compute the spectral centroid of an audio frame.

        Parameters:
        frame (Frame): A Frame object.

        Returns:
        float: The spectral centroid of the frame.
        """

        # Calculate the FFT and magnitude spectrum
        fft = np.fft.fft(frame.samples)
        magnitude_spectrum = np.abs(fft)

        # Compute the frequency values for each FFT bin
        n = len(frame.samples)
        freqs = np.fft.fftfreq(n, d=1 / frame.duration)[:n // 2]

        # Calculate the spectral centroid
        centroid = np.sum(freqs * magnitude_spectrum[:n // 2]) / np.sum(magnitude_spectrum[:n // 2])

        return centroid

    @staticmethod
    def spectral_centroid_std(frames: List[Frame]) -> float:
        """
        Compute the standard deviation of spectral centroids for a list of frames.

        Parameters:
        frames (List[Frame]): A list of Frame objects.

        Returns:
        float: The standard deviation of spectral centroids.
        """

        spectral_centroids = [App.spectral_centroid(frame) for frame in frames]
        return np.std(spectral_centroids)
    
    @staticmethod
    def spectral_bandwidth(frame: Frame) -> float:
        """
        Compute the spectral bandwidth of an audio frame.

        Parameters:
        frame (Frame): A Frame object.

        Returns:
        float: The spectral bandwidth of the frame.
        """

        # Calculate the FFT and magnitude spectrum
        fft = np.fft.fft(frame.samples)
        magnitude_spectrum = np.abs(fft)

        # Compute the frequency values for each FFT bin
        n = len(frame.samples)
        freqs = np.fft.fftfreq(n, d=1 / frame.duration)[:n // 2]

        # Calculate the spectral centroid
        spectral_centroid = App.spectral_centroid(frame)

        # Calculate the spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum((freqs - spectral_centroid)**2 * magnitude_spectrum[:n // 2]) / np.sum(magnitude_spectrum[:n // 2]))

        return spectral_bandwidth


    @staticmethod
    def spectral_contrast(frame: Frame, sampling_rate: int, n_bands: int = 4) -> np.ndarray:
        # Calculate the FFT and magnitude spectrum
        fft = np.fft.fft(frame.samples)
        magnitude_spectrum = np.abs(fft)[: len(fft) // 2]

        # Divide the frequency range into subbands
        freqs = np.linspace(0, sampling_rate // 2, len(magnitude_spectrum), endpoint=False)
        indices = np.arange(len(magnitude_spectrum))
        subbands = np.array_split(indices, n_bands)

        # Compute the spectral contrast for each subband
        spectral_contrast = []
        

        for band in subbands:
            max_val = np.max(magnitude_spectrum[band])
            min_val = np.min(magnitude_spectrum[band])
            contrast = max_val / min_val
            spectral_contrast.append(contrast)

        return np.array(spectral_contrast)


    @staticmethod
    def spectral_contrast_std(frames: List[Frame], n_bands: int = 6) -> np.ndarray:
        """
        Compute the standard deviation of spectral contrast for a list of frames.

        Parameters:
        frames (List[Frame]): A list of Frame objects.
        n_bands (int): Number of frequency subbands to divide the spectrum into.

        Returns:
        np.ndarray: The standard deviation of spectral contrast values for each subband.
        """

        spectral_contrasts = np.array([App.spectral_contrast(frame, n_bands) for frame in frames])
        return np.std(spectral_contrasts, axis=0)
    

    @staticmethod
    def plot_spectral_contrast_std(spectral_contrast_std: np.ndarray, n_bands: int = 6):
        """
        Plot the standard deviation of spectral contrast values for each frequency subband.

        Parameters:
        spectral_contrast_std (np.ndarray): The standard deviation of spectral contrast values.
        n_bands (int): Number of frequency subbands.
        """

        # Create a bar plot
        subband_labels = [f"Band {i + 1}" for i in range(n_bands)]
        fig = go.Figure([go.Bar(x=subband_labels, y=spectral_contrast_std)])

        # Set plot title and labels
        fig.update_layout(
            title="Standard Deviation of Spectral Contrast",
            xaxis_title="Frequency Subbands",
            yaxis_title="Standard Deviation",
            template="plotly_white"
        )

        return fig


    @staticmethod
    def spectral_bandwidth_std(frames: List[Frame]) -> float:
        """
        Compute the standard deviation of spectral bandwidths for a list of frames.

        Parameters:
        frames (List[Frame]): A list of Frame objects.

        Returns:
        float: The standard deviation of spectral bandwidths.
        """

        spectral_bandwidths = [App.spectral_bandwidth(frame) for frame in frames]
        return np.std(spectral_bandwidths)


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
            yaxis_range=[-1.05, 1.05],
            **self._DEFAULT_PARAMS,
        )
        fig.update_traces(line_color="#16733e")
        return fig
    
    def plot_sample_with_SR(
            self, 
            frame_duration_miliseconds: int = 10, 
            vol_min_value=0.0, 
            vol_max_value=0.05, 
            zcr_min_value=0.5, 
            zcr_max_value=2
        ):
        fig = self.plot_sample()
        frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]
        sr_list = [
            self.SR(frame, vol_min_value, vol_max_value, zcr_min_value, zcr_max_value) 
                for frame in self.frame_generator(frame_duration_miliseconds)]
        idx = 0
        rect_start_idx = None
        for frame, sr in zip(frames, sr_list):
            if rect_start_idx is None and sr:
                rect_start_idx = idx
            if rect_start_idx is not None and not sr:
                fig.add_shape(
                    type="rect", 
                    x0=rect_start_idx,
                    x1=idx + len(frame.samples),
                    y0=-1,
                    y1=1,
                    fillcolor="#e63946", 
                    opacity=0.2,
                    line_width=0
                )
                rect_start_idx = None
            idx += len(frame.samples)

        return fig
        
    def plot_frame_level_feature(
            self, 
            frame_level_func: Callable[[Frame], float], 
            plot_title: str,
            frame_duration_miliseconds: int = 10,
            min_val: float = None, 
            max_val: float = None,
            fig_layout_kwargs={}
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
            rect_start_idx = None
            for i in range(len(y)-1):
                if rect_start_idx is None and min_val <= y[i] <= max_val:
                    rect_start_idx = i
                if rect_start_idx is not None and not (min_val <= y[i] <= max_val):
                    fig.add_shape(
                        type="rect", x0=x[rect_start_idx], y0=min_val, x1=x[i], y1=max_val,
                        fillcolor="#e63946", opacity=0.2, line_width=0
                    )
                    rect_start_idx = None
            if rect_start_idx is not None:
                fig.add_shape(
                    type="rect", x0=x[rect_start_idx], y0=min_val, x1=x[-1], y1=max_val,
                    fillcolor="#e63946", opacity=0.2, line_width=0
                )

        fig.update_layout(
            xaxis=self._XAXIS_PARAMS,
            yaxis=self._YAXIS_PARAMS,
            yaxis_title=None,
            title=plot_title,
            title_x=0.05,
            title_y=0.95,
            **{
                **self._DEFAULT_PARAMS,
                **fig_layout_kwargs
            }
        )

        return fig


    def plot_autocorrelation(
        self,
        frame_level_func: Callable[[Frame], Tuple[np.ndarray, float]],
        frame_numbers: List[int],
        plot_title: str,
        frame_duration_miliseconds: int = 10,
        fig_layout_kwargs={},
    ):

        frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]
        selected_frames = [frames[i - 1] for i in frame_numbers]

        color_list = ['#16733e', '#e63946', '#0077b6', '#9c89b8', '#f48c06']

        fig = go.Figure()

        for idx, frame in enumerate(selected_frames):
            x = np.arange(len(frame.samples)) / frame_level_func(frame)[1] * 1000
            y = frame_level_func(frame)[0]
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", line=dict(color=color_list[idx % len(color_list)]), showlegend=False)
            )

        fig.update_layout(
            xaxis={
                "tickmode": "linear",
                "tick0": 0,
                "dtick": 50,
                "linecolor": "black",
                "gridcolor": "#c4cfc9",
                "showline": True,
                "mirror": True,
                "ticks": "outside",
                "title": "Time [ms]"
            },
            yaxis=self._YAXIS_PARAMS,
            yaxis_title=None,
            title=plot_title,
            title_x=0.05,
            title_y=0.95,
            **{
                **self._DEFAULT_PARAMS,
                **fig_layout_kwargs
            }
        )

        return fig
    

    def plot_spectrogram(self, 
                         n_fft: int = 2048, 
                         hop_length: int = 512,
                        cmap: str = "viridis"):
        """
        Function to plot the spectrogram of the audio signal.

        Parameters:
        n_fft (int): The number of samples in the FFT window. Default is 2048.
        hop_length (int): The number of samples between successive frames. Default is 512.
        cmap (str): The color map to use for the plot. Default is "viridis".

        Returns:
        plotly.graph_objs.Figure: A plotly Figure object containing the spectrogram plot.
        """

        # Compute the spectrogram
        spectrogram = np.abs(librosa.stft(self.samples, n_fft=n_fft, hop_length=hop_length))
        log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

        # Create the plot
        fig = px.imshow(
            log_spectrogram,
            x=np.arange(0, len(self.samples) / self.frame_rate, hop_length / self.frame_rate),
            y=librosa.fft_frequencies(sr=self.frame_rate, n_fft=n_fft),
            labels=dict(x="Time [s]", y="Frequency [Hz]", color="Amplitude (dB)"),
            origin="lower",
            aspect="auto",
            color_continuous_scale=cmap,
            template="plotly_white",
        )

        fig.update_layout(
            xaxis=self._XAXIS_PARAMS,
            yaxis=self._YAXIS_PARAMS,
            yaxis_title="Frequency [Hz]",
            title="Spectrogram",
            title_x=0.05,
            title_y=0.95,
            **self._DEFAULT_PARAMS,
        )

        return fig


    def get_frame_level_export_data(self, frame_duration_miliseconds):
        records = {}

        frame_funcs = {
            "Volume": self.volume,
            "Short-Time Energy": self.STE,
            "Zero Crossing Rate": self.ZCR,
            "Silent ratio": self.SR,
            "Max autocorrelation": self.autocorrelation,
            "RMS": self.RMS,
        }

        frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]

        for name, func in frame_funcs.items():
            if name == "Max autocorrelation":
                autocorr_max = []
                for frame in frames:
                    autocorr, sample_rate = func(frame)
                    autocorr_max.append(autocorr.max())
                records[name] = autocorr_max
            else:
                records[name] = [func(frame) for frame in frames]

        return pd.DataFrame(records).round(4)

    def get_clip_level_export_data(self, frame_duration_miliseconds=10):
        records = {}

        frame_funcs = {
            "VSTD": self.VSTD,
            "VDR": self.VDR,
            "VU": self.VU,
            "LSTER": self.LSTER,
            "ENERGY ENTROPY": self.energy_entropy,
            "HZCRR": self.HZCRR,
        }

        frames = [frame for frame in self.frame_generator(frame_duration_miliseconds)]

        for name, func in frame_funcs.items():
            records[name] = func(frames)
        return pd.DataFrame([records], index=[0]).round(4)
    
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


    def plot_music_vs_talks(self, lster_list: List[float],
                        threshold_lster: float,
                        plot_title: str):
        music_x, music_y = [], []
        talks_x, talks_y = [], []
        
        for i in range(len(lster_list)):
            if lster_list[i] <= threshold_lster:
                music_x.append(i)
                music_y.append(1)
                talks_x.append(i)
                talks_y.append(0)
            else:
                music_x.append(i)
                music_y.append(0)
                talks_x.append(i)
                talks_y.append(1)
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=music_x, y=music_y, mode='lines', line=dict(color="#16733e"), name='Muzyka'))
        fig.add_trace(go.Scatter(x=talks_x, y=talks_y, mode='lines', line=dict(color="#ffbe0b"), name='mowa'))

        fig.update_layout(
            xaxis_title="Segment dźwięku",
            yaxis_title="LSTER",
            title=plot_title,
            yaxis_range=[0, 1]
        )

        return fig