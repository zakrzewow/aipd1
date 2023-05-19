import streamlit as st
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Określanie fragmentów dźwiecznych i bezdźwięcznych")

    st.subheader("Fragment bezdźwięczny - ustawienia")
    st.write(
        "Ramka jest uznawana za fragment bezdźwięczny \
                jeśli wartości STE oraz ZCR znajdują się w żądanych przedziałach"
    )

    frame_duration = st.slider(
        key="slider_page_5",
        label="Długość ramki (w milisekundach):",
        min_value=5,
        max_value=40,
        value=25,
        step=5,
    )

    col_vol, col_zcr = st.columns(2)
    with col_vol:
        st.markdown("#### STE")
        col_vol_min, col_vol_max = st.columns(2)
        ste_min_val = col_vol_min.number_input(
            key="ste_min_np_page4",
            label="STE MIN",
            min_value=0.00,
            max_value=2.00,
            value=0.000,
            step=0.005,
            format="%.3f",
        )

        ste_max_val = col_vol_max.number_input(
            key="ste_max_np_page4",
            label=" STE MAX",
            min_value=0.00,
            max_value=2.00,
            value=0.01,
            step=0.005,
            format="%.3f",
        )
        # volume
        plot_volume = app.plot_frame_level_feature(
            frame_level_func=app.STE,
            plot_title="",
            frame_duration_miliseconds=frame_duration,
            min_val=ste_min_val,
            max_val=ste_max_val,
            fig_layout_kwargs=dict(width=500, height=300),
        )
        st.plotly_chart(plot_volume)

    with col_zcr:
        st.markdown("#### ZCR")
        col_zcr_min, col_zcr_max = st.columns(2)
        zcr_min_val = col_zcr_min.number_input(
            key="zcr_min_np_page_1",
            label="ZCR MIN",
            min_value=0.0,
            max_value=2.0,
            value=0.4,
            step=0.05,
        )

        zcr_max_val = col_zcr_max.number_input(
            key="zcr_max_np_page_2",
            label="ZCR MAX",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.05,
        )
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

    st.subheader("Zaznaczenie fragmentów bezdźwięcznych na wykresie")
    plot_sample_with_sr = app.plot_sample_with_SR(
        frame_duration, ste_min_val, ste_max_val, zcr_min_val, zcr_max_val
    )
    st.plotly_chart(plot_sample_with_sr)
