import streamlit as st
import pandas as pd
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Wykresy sygnału w dziedzinie czasu i częstotliwości")

    sample_start, sample_end = st.columns(2)
    max_ = len(app.samples)
    sample_start_val = sample_start.number_input("Początek sygnału", 0, max_, 0, 1)
    sample_end_val = sample_end.number_input("Konies sygnału", 0, max_, max_, 1)

    available_window_names = [
        "okno prostokątne",
        "okno trójkątne",
        "boxcar",
        "hann",
        "hamming",
        "blackman",
        "bartlett",
    ]
    selected_window_name = st.radio(
        label="Wybierz funkcję okienkową:",
        options=available_window_names,
        index=0,  # Default selected option
    )

    time_domain_spectrum_plot = frequencyApp.plot_time_domain_spectrum_windows(
        window_name=selected_window_name,
        sample_start=sample_start_val,
        sample_end=sample_end_val,
    )
    st.plotly_chart(time_domain_spectrum_plot)

    spectrum_plot = frequencyApp.plot_spectrum_windows(
        window_name=selected_window_name,
        sample_start=sample_start_val,
        sample_end=sample_end_val,
    )
    st.plotly_chart(spectrum_plot)
