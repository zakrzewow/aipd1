import streamlit as st
import pandas as pd
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Spektogram")
    # Create a window type selector
    window_options = [
        "okno prostokątne",
        "okno trójkątne",
        "boxcar",
        "hann",
        "hamming",
        "blackman",
        "bartlett",
    ]
    selected_window = st.selectbox("Wybierz funkcję okienkową", window_options)
    print(window_options)

    # Create sliders for NFFT and noverlap
    NFFT = st.slider("Długość ramki", 128, 4096, 1024, step=128)
    noverlap = st.slider(
        "Długość nakładania ramek", 0, NFFT - 128, NFFT // 2, step=128
    )

    # Call the create_spectrogram and plot_spectrogram methods
    frequencyApp.create_spectrogram(NFFT, selected_window, noverlap)
    spectrogram_plot = frequencyApp.plot_spectrogram(
        NFFT, selected_window, noverlap
    )

    # Display the spectrogram
    st.plotly_chart(spectrogram_plot)
