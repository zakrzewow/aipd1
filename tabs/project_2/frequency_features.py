import streamlit as st
import pandas as pd
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Parametry dźwieku z dziedziny częstotliwości")

    frame_duration_frec = st.slider(
        key="slider_frec",
        label="Długość ramki (w milisekundach):",
        min_value=10,
        max_value=40,
        value=25,
        step=5,
    )

    # volume
    plot_volume_frec = frequencyApp.plot_frame_level_feature(
        frame_level_func=frequencyApp.volume,
        plot_title="VOLUME",
        frame_duration_miliseconds=frame_duration_frec,
    )
    st.plotly_chart(plot_volume_frec)

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
