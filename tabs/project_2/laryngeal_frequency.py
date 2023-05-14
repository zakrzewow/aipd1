import streamlit as st
import pandas as pd
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Częstotliwośc krtaniowa")
    st.subheader("Cepstrum")
    cepstrum_signal_plot = frequencyApp.visualize_cepstrum_signal()
    st.plotly_chart(cepstrum_signal_plot)
    st.subheader("Czestotliwość kratniowa (F0)")
    window_size = st.slider(
        "Rozmiar okna", min_value=256, max_value=2 ** 13, value=256, step=128
    )
    step_size = st.slider(
        "Długośc nakładania ramek",
        min_value=256,
        max_value=2 ** 13,
        value=256,
        step=128,
    )

    window_options = [
        "okno prostokątne",
        "okno trójkątne",
        "boxcar",
        "hann",
        "hamming",
        "blackman",
        "bartlett",
    ]
    selected_window_function = st.selectbox("Wybierz funkcję okienkową", window_options)

    # Use the selected window function when calling visualize_laryngeal_frequency
    f0_signal_plot = frequencyApp.visualize_laryngeal_frequency(
        window_size, step_size, selected_window_function
    )
    st.plotly_chart(f0_signal_plot)

    st.subheader("Czestotliość tonu podstawowego dla całego klipu wynosi:")
    st.markdown(
        f"**{frequencyApp.laryngeal_frequency(window_function = selected_window_function)[1][0]}Hz**"
    )
