import streamlit as st
import pandas as pd
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Wykresy sygnału w dziedzinie czasu i częstotliwości")

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
        window_name=selected_window_name
    )
    st.plotly_chart(time_domain_spectrum_plot)

    spectrum_plot = frequencyApp.plot_spectrum_windows(window_name=selected_window_name)
    st.plotly_chart(spectrum_plot)
