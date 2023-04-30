import streamlit as st
import pandas as pd
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Analiza na poziomie klipu")

    # Define the column names for the table
    col1 = "Frame-Level Function"
    col2 = "Parameter"

    # Define the default values for the table
    frame_level_functions = [
        "Volume",
        "Short-Time Energy",
        "Zero Crossing Rate",
        "Silent ratio",
    ]
    clip_level_functions = ["VSTD", "VDR", "VU", "LSTER", "ENERGY ENTROPY", "HZCRR"]
    all_functions = frame_level_functions + clip_level_functions

    data = pd.DataFrame(
        {
            col1: all_functions,
            col2: [0.5] * len(all_functions),
        }  # Change this list to set default parameter values
    )

    # Create the table
    table = st.experimental_data_editor(data, num_rows="dynamic")

    st.download_button(
        label="Pobierz (.csv)",
        data=table.to_csv(),
        file_name=f"table.csv",
        mime="text/csv",
    )
