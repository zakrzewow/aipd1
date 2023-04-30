import streamlit as st
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Przebieg czasowy pliku audio")
    plot_sample = app.plot_sample()
    st.plotly_chart(plot_sample)
