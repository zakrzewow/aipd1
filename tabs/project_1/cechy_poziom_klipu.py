import streamlit as st
import pandas as pd
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Analiza na poziomie klipu")

    vstd = app.VSTD(app.frames)
    vdr = app.VDR(app.frames)
    vu = app.VU(app.frames)
    lster = app.LSTER(app.frames)
    energy_entropy = app.energy_entropy(app.frames)
    hzcrr = app.HZCRR(app.frames)
    spectral_centroid = app.spectral_centroid_std(app.frames)
    spectral_bandwidth = app.spectral_bandwidth_std(app.frames)

    data = {
        "Values": [
            vstd,
            vdr,
            vu,
            lster,
            energy_entropy,
            hzcrr,
            spectral_centroid,
            spectral_bandwidth,
        ]
    }

    # Create DataFrame
    df = pd.DataFrame(
        data,
        index=[
            "VSTD",
            "VDR",
            "VU",
            "LSTER",
            "ENERGY ENTROPY",
            "HZCRR",
            "SPECTRAL CENTROID",
            "SPECTRAL BANDWIDTH",
        ],
    )

    # Select only one column
    df_column = df[["Values"]]

    # Rename index
    df_column.index.name = "Metrics"

    # Display table in Streamlit
    st.write(df_column)

    # Plot the standard deviation of spectral contrast values
    spectral_contrast_std_values = app.spectral_contrast_std(app.frames)
    if len(spectral_contrast_std_values) == 0:
        st.write("Audio jest na za którkie na analizę Spectral Contrast Std")
    else:
        plot_spectral_contrast_std = app.plot_spectral_contrast_std(
            spectral_contrast_std_values
        )
        st.plotly_chart(plot_spectral_contrast_std)

    # Plot spectrogram
    st.subheader("Spectrogram")
    spectrogram = app.plot_spectrogram()
    st.plotly_chart(spectrogram)

    ## Download data
    st.subheader("Zapis parametrów")
    clip_level_data = app.get_clip_level_export_data()
    st.download_button(
        label="Pobierz parametry (.csv)",
        data=clip_level_data.to_csv(),
        file_name=f"{file_name.split('.')[0]}_frame_10.csv",
        mime="text/csv",
    )
