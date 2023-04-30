import streamlit as st
from modules.timeApp import TimeApp


def run(app: TimeApp):
    st.header("Przebieg czasowy pliku audio")
    plot_sample = app.plot_sample()
    st.plotly_chart(plot_sample)

