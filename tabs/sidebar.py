import os
import streamlit as st
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp
from typing import Tuple, Dict


def run(TABS: Dict) -> Tuple[str, str, TimeApp, FrequencyApp]:
    st.sidebar.subheader("Wgraj własny plik .wav")
    uploaded_file = st.sidebar.file_uploader(
        "file_selectbox", "wav", label_visibility="collapsed"
    )

    st.sidebar.header("LUB")
    st.sidebar.subheader("Wybierz głos")
    users = ["Kacper", "Grzegorz", "głos kobiecy"]
    user = st.sidebar.selectbox(
        "user_selectbox",
        users,
        label_visibility="collapsed",
        disabled=uploaded_file is not None,
    )

    directory = f"./samples/{user}/Znormalizowane/"
    file_list = os.listdir(directory)
    st.sidebar.subheader("Wybierz plik")
    file_name = st.sidebar.selectbox(
        "file_selectbox",
        file_list,
        label_visibility="collapsed",
        disabled=uploaded_file is not None,
    )

    if uploaded_file is None:
        file_path = os.path.join(directory, file_name)
        app = TimeApp(file_path)
        frequencyApp = FrequencyApp(file_path)
    else:
        app = TimeApp(uploaded_file)
        frequencyApp = FrequencyApp(file_path)

    # audio
    st.sidebar.audio(app.samples, sample_rate=app.frame_rate)

    # tabs

    # "Wykresy sygnału w dziedzinie czasu i częstotliwości",
    tabs = [*TABS.keys(), "Informacje"]
    selected_tab = st.sidebar.radio("tab_radio", tabs, label_visibility="collapsed")

    return selected_tab, file_name, app, frequencyApp
