import numpy as np
import plotly.graph_objs as go
from pydub import AudioSegment
from modules.frame import Frame
from typing import Callable
from scipy.signal import get_window
from modules.app import App

class FrequencyApp(App):
    # stylowanie wykresów

    def read_wav(self, filepath_or_bytes, normalize=True):
        super().read_wav(filepath_or_bytes, normalize=normalize)
        self.samples = self.to_next_2n(self.samples)

    def to_next_2n(self, samples):
        """
        Adds zeroes in the end of the signal to achieve the length of 2^n, where n
        is a positive integer. Required for FFT.

        Parameters
        ----------
        samples : vector of amplitudes (samples / frames[i])

        Returns
        ----------
        new_samples : vector of samples of length 2^n with zeroes added
        """  
        k = len(samples)
        n = 1
        while k > n:
            n = 2*n
        m = n-k
        new_samples = np.append(samples, np.asarray([0]*m))
        return new_samples

    @staticmethod
    def rectangular_window(M):
        return np.ones(M)

    @staticmethod
    def triangular_window(M):
        return np.bartlett(M)

    @staticmethod
    def volume(frame: Frame):
        spectrum = np.fft.fft(frame.samples)
        magnitude_spectrum = np.abs(spectrum)
        volume = np.sum(np.power(magnitude_spectrum, 2)) / len(frame.samples)
        return volume
    
    @staticmethod
    def window_rfft(samples, window_function):
        if callable(window_function):
            window = window_function(len(samples))
        else:
            window = get_window(window_function, len(samples))
        windowed_signal = samples * window
        rfft_output = np.fft.rfft(windowed_signal)
        return rfft_output
    
    def real_cepstrum_signal(self):
        fft = np.fft.fft(self.samples)
        log_fft = np.log10(np.abs(fft))
        cep_sig = np.fft.irfft(log_fft)
        return cep_sig

    def estimate_fundamental_frequency(self, samples):
        spectrum = np.fft.rfft(samples)
        log_amplitude_spectrum = np.log(np.abs(spectrum))
        cepstrum = np.fft.rfft(log_amplitude_spectrum)
        real_cepstrum = np.abs(cepstrum)

        lower_freq = 50
        upper_freq = 400
        lower_tau = int(self.frame_rate / upper_freq)
        upper_tau = int(self.frame_rate / lower_freq)

        max_tau = np.argmax(real_cepstrum[lower_tau:upper_tau]) + lower_tau
        fundamental_frequency = self.frame_rate / max_tau

        return fundamental_frequency
    
    def short_time_energy(self, window_size=1024, step_size=512, window_function="hann"):
        num_windows = (len(self.samples) - window_size) // step_size + 1
        windows = [self.samples[i * step_size:i * step_size + window_size] for i in range(num_windows)]

        window_function_map = {
            "okno prostokątne": self.rectangular_window,
            "okno trójkątne": self.triangular_window,
            "boxcar": "boxcar",
            "hann": "hann",
            "hamming": "hamming",
            "blackman": "blackman",
            "bartlett": "bartlett",
        }
        window_func = window_function_map[window_function]

        ste = []
        for w in windows:
            windowed_signal = w * window_func(len(w))
            energy = np.sum(windowed_signal**2)
            ste.append(energy)
        return ste

    def laryngeal_frequency(self, window_size=None, step_size=None, window_function="hann"):
        if window_size is None:
            window_size = len(self.samples)
        if step_size is None:
            step_size = window_size * 2  # Arbitrary large number to ensure only one window is created

        num_windows = (len(self.samples) - window_size) // step_size + 1
        windows = [self.samples[i * step_size:i * step_size + window_size] for i in range(num_windows)]

        ste = self.short_time_energy(window_size, step_size, window_function)
        energy_threshold = np.mean(ste) * 0.2  # You may need to adjust this multiplier based on your data

        lf = []
        for i, window in enumerate(windows):
            if ste[i] >= energy_threshold:
                lf.append(self.estimate_fundamental_frequency(window))
            else:
                lf.append(None)  # Use None to indicate unvoiced segments
        xaxes = np.arange(len(lf))
        return xaxes, lf

    def visualize_laryngeal_frequency(self, window_size=1024, step_size=512, window_function="hann"):
        xaxes, lf = self.laryngeal_frequency(window_size, step_size, window_function)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xaxes, y=lf, mode='markers', marker=dict(color='#16733e')))
        fig.update_layout(
            title="Laryngeal Frequency",
            xaxis_title="Window Number",
            yaxis_title="Fundamental Frequency (Hz)",
        )

        return fig
 
    def visualize_cepstrum_signal(self):
        cepstrum = self.real_cepstrum_signal()
        quefrencies = np.array(range(len(self.samples))) / self.frame_rate

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=quefrencies, y=cepstrum, mode='lines', line=dict(color='#16733e')))
        fig.update_layout(
            title="Cepstrum real signal",
            xaxis_title="Quefrency (s)",
            yaxis_title="Absolute Cepstrum",
            yaxis=self._YAXIS_PARAMS,
            **self._DEFAULT_PARAMS,
        )
        return fig

    def plot_spectrum_windows(self, window_name: str ):
        window_function_map = {
            "okno prostokątne": self.rectangular_window,
            "okno trójkątne": self.triangular_window,
            "boxcar": "boxcar",
            "hann": "hann",
            "hamming": "hamming",
            "blackman": "blackman",
            "bartlett": "bartlett",
        }

        f = np.fft.rfftfreq(len(self.samples), 1/self.frame_rate)
        window_function = window_function_map[window_name]
        spectrum_complex = self.window_rfft(self.samples, window_function)
        spectrum = 20 * np.log10(np.abs(spectrum_complex))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=spectrum, mode='lines', line=dict(color='#16733e')))
        fig.update_layout(
            title='Widmo decybelowe sygnału z wybranym oknem',
            xaxis_title='Częstotliwość [Hz]',
            yaxis_title='Amplituda widma [dB]',
            yaxis=self._YAXIS_PARAMS,
            **self._DEFAULT_PARAMS,
        )
        return fig
    

    def plot_time_domain_spectrum_windows(self, window_name: str):
        window_function_map = {
            "okno prostokątne": self.rectangular_window,
            "okno trójkątne": self.triangular_window,
            "boxcar": "boxcar",
            "hann": "hann",
            "hamming": "hamming",
            "blackman": "blackman",
            "bartlett": "bartlett",
        }

        # Same as before
        window_function = window_function_map[window_name]
        spectrum_complex = self.window_rfft(self.samples, window_function)
        
        # Apply the inverse Fourier transform to the complex spectrum
        time_domain_signal = np.fft.irfft(spectrum_complex)

        # Create the time values for the x-axis
        time_values = np.linspace(0, len(self.samples) / self.frame_rate, len(time_domain_signal))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_values, y=time_domain_signal, mode='lines', line=dict(color='#16733e')))
        fig.update_layout(
            title='Sygnał w dziedzinie czasu z wybranym oknem (w dB)',
            xaxis_title='Czas [s]',
            yaxis_title='Amplituda',
            yaxis=self._YAXIS_PARAMS,
            **self._DEFAULT_PARAMS,
        )

        return fig
    
    def create_spectrogram(self, NFFT, window_name, noverlap=None):
        ts = self.samples
        if noverlap is None:
            noverlap = NFFT // 2
        noverlap = int(noverlap)
        starts = np.arange(0, len(ts), NFFT - noverlap, dtype=int)
        starts = starts[starts + NFFT < len(ts)]
        
        window_function_map = {
            "okno prostokątne": self.rectangular_window,
            "okno trójkątne": self.triangular_window,
            "boxcar": "boxcar",
            "hann": "hann",
            "hamming": "hamming",
            "blackman": "blackman",
            "bartlett": "bartlett",
        }
        
        window_function = window_function_map[window_name]
        xns = []
        
        for start in starts:
            ts_segment = ts[start:start + NFFT]
            if callable(window_function):
                window = window_function( len(ts_segment))
            else:
                window = get_window(window_function, len(ts_segment))
            windowed_ts = ts_segment * window
            ts_window = np.fft.rfft(windowed_ts)
            xns.append(ts_window)
        
        
        specX = np.array(xns).T
        spec = 10 * np.log10(np.abs(specX))
        assert spec.shape[1] == len(starts)
        return starts, spec
    

    def plot_spectrogram(self, NFFT, window_name, noverlap=None):
        starts, spec = self.create_spectrogram(NFFT, window_name, noverlap)
        sample_rate = self.frame_rate
        samples = self.samples
        L = NFFT

        # Convert to frequency in kHz and time in seconds
        freqs = np.linspace(0, sample_rate / 2 / 1000, spec.shape[0])
        times = np.linspace(0, len(samples) / sample_rate, spec.shape[1])

        fig = go.Figure(go.Heatmap(
            z=spec,
            x=times,
            y=freqs,
            colorscale='Viridis',
            zmin=np.min(spec),
            zmax=np.max(spec)
        ))

        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Frequency (kHz)",
            template='plotly_dark'
        )

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
        x = [frame_number for frame_number in range(1, len(frames)+1,1)]
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

