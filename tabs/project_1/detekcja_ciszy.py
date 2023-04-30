import streamlit as st
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Detekcja ciszy")

    st.subheader("Silent Ratio - ustawienia")
    st.write(
        "Ramka jest uznawana za ciszę jeśli wartości Volume oraz ZCR znajdują się w żądanym przedziale."
    )

    frame_duration = st.slider(
        key="slider_page_3",
        label="Długość ramki (w milisekundach):",
        min_value=10,
        max_value=40,
        value=25,
        step=5,
    )

    col_vol, col_zcr = st.columns(2)
    with col_vol:
        st.markdown("#### Volume")
        col_vol_min, col_vol_max = st.columns(2)
        vol_min_val = col_vol_min.number_input("Volume MIN", 0.0, 0.5, 0.0, 0.01)
        vol_max_val = col_vol_max.number_input("Volume MAX", 0.0, 0.5, 0.05, 0.01)
        # volume
        plot_volume = app.plot_frame_level_feature(
            frame_level_func=app.volume,
            plot_title="",
            frame_duration_miliseconds=frame_duration,
            min_val=vol_min_val,
            max_val=vol_max_val,
            fig_layout_kwargs=dict(width=500, height=300),
        )
        st.plotly_chart(plot_volume)

    with col_zcr:
        st.markdown("#### ZCR")
        col_zcr_min, col_zcr_max = st.columns(2)
        zcr_min_val = col_zcr_min.number_input("ZCR MIN", 0.0, 2.0, 0.5, 0.05)
        zcr_max_val = col_zcr_max.number_input("ZCR MAX", 0.0, 2.0, 2.0, 0.05)
        # zcr
        plot_zcr = app.plot_frame_level_feature(
            frame_level_func=app.ZCR,
            plot_title="",
            frame_duration_miliseconds=frame_duration,
            min_val=zcr_min_val,
            max_val=zcr_max_val,
            fig_layout_kwargs=dict(width=500, height=300),
        )
        st.plotly_chart(plot_zcr)

    st.subheader("Zaznaczenie ciszy na wykresie")
    plot_sample_with_sr = app.plot_sample_with_SR(
        frame_duration, vol_min_val, vol_max_val, zcr_min_val, zcr_max_val
    )
    st.plotly_chart(plot_sample_with_sr)
