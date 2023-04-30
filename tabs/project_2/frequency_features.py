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
        frame_level_func=frequencyApp.Volume,
        plot_title="VOLUME",
        frame_duration_miliseconds=frame_duration_frec,
    )
    st.plotly_chart(plot_volume_frec)

    # frequency centroid
    plot_FC = frequencyApp.plot_frame_level_feature(
        frame_level_func=frequencyApp.FrequencyCentroid,
        plot_title="Frequency Centroid",
        frame_duration_miliseconds=frame_duration_frec,
    )
    st.plotly_chart(plot_FC)

    # band energy & band energy ratio
    frequency_bands = [(0, 630), (630, 1720), (1720, 4400)]
    frequency_bands_names = ["0 - 630 Hz", "630 - 1720 Hz", "1720 - 4400 Hz"]
    bands_tabs = st.tabs(frequency_bands_names)

    for frequency_band, band_tab in zip(frequency_bands, bands_tabs):
        plot_BE = frequencyApp.plot_frame_level_feature(
            frame_level_func=frequencyApp.BandEnergy,
            plot_title=f"Band Energy {frequency_band[0]} - {frequency_band[1]} Hz",
            frame_duration_miliseconds=frame_duration_frec,
            frame_level_func_kwargs=dict(frequency_band=frequency_band),
        )
        band_tab.plotly_chart(plot_BE)

        plot_ERSB = frequencyApp.plot_frame_level_feature(
            frame_level_func=frequencyApp.BandEnergyRatio,
            plot_title=f"Band Energy Ratio {frequency_band[0]} - {frequency_band[1]} Hz",
            frame_duration_miliseconds=frame_duration_frec,
            frame_level_func_kwargs=dict(frequency_band=frequency_band),
        )
        band_tab.plotly_chart(plot_ERSB)

    # spectral flatness
    plot_SFM = frequencyApp.plot_frame_level_feature(
        frame_level_func=frequencyApp.SpectralFlatnessMeasure,
        plot_title="Spectral Flatness Measure",
        frame_duration_miliseconds=frame_duration_frec,
    )
    st.plotly_chart(plot_SFM)

    # spectral flatness
    plot_SCF = frequencyApp.plot_frame_level_feature(
        frame_level_func=frequencyApp.SpectralCrestFactor,
        plot_title="Spectral Crest Factor",
        frame_duration_miliseconds=frame_duration_frec,
    )
    st.plotly_chart(plot_SCF)
