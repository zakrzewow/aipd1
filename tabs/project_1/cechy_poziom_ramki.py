import streamlit as st
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Cechy sygnału audio na poziomie ramki")

    frame_duration = st.slider(
        key="slider_page_1",
        label="Długość ramki (w milisekundach):",
        min_value=10,
        max_value=40,
        value=25,
        step=5,
    )

    st.subheader("Wykresy")

    # volume
    plot_volume = app.plot_frame_level_feature(
        frame_level_func=app.volume,
        plot_title="VOLUME",
        frame_duration_miliseconds=frame_duration,
    )
    st.plotly_chart(plot_volume)

    # ste
    plot_ste = app.plot_frame_level_feature(
        frame_level_func=app.STE,
        plot_title="STE",
        frame_duration_miliseconds=frame_duration,
    )
    st.plotly_chart(plot_ste)

    # zcr
    plot_zcr = app.plot_frame_level_feature(
        frame_level_func=app.ZCR,
        plot_title="ZCR",
        frame_duration_miliseconds=frame_duration,
    )
    st.plotly_chart(plot_zcr)

    # F0 corr
    plot_f0_cor = app.plot_frame_level_feature(
        frame_level_func=app.F0_Cor,
        plot_title="F0 Corr",
        frame_duration_miliseconds=frame_duration,
    )
    st.plotly_chart(plot_f0_cor)

    # F0 MADF
    plot_f0_amdf = app.plot_frame_level_feature(
        frame_level_func=app.F0_AMDF,
        plot_title="F0 AMDF",
        frame_duration_miliseconds=frame_duration,
    )
    st.plotly_chart(plot_f0_amdf)

    # autocorrealation
    frame_numbers_input = st.text_input(
        "Enter frame numbers to plot separated by commas (e.g. 1,2,3):", "1,2,3"
    )
    frame_numbers = [int(x.strip()) for x in frame_numbers_input.split(",")]
    plot_autocorr = app.plot_autocorrelation(
        frame_level_func=app.autocorrelation,
        frame_numbers=frame_numbers,
        plot_title="Autocorrealtion",
        frame_duration_miliseconds=frame_duration,
    )
    st.plotly_chart(plot_autocorr)

    # RMS
    plot_rms = app.plot_frame_level_feature(
        frame_level_func=app.RMS,
        plot_title="RMS",
        frame_duration_miliseconds=frame_duration,
    )
    st.plotly_chart(plot_rms)

    ## Download data
    st.subheader("Zapis parametrów")
    frame_level_data = app.get_frame_level_export_data(
        frame_duration_miliseconds=frame_duration
    )
    st.download_button(
        label="Pobierz parametry (.csv)",
        data=frame_level_data.to_csv(),
        file_name=f"{file_name.split('.')[0]}_frame_{frame_duration}.csv",
        mime="text/csv",
    )

    ## Define a dropdown to allow the user to select the frame-level function
    st.subheader("Odsłuchaj ramki z nagrania spełniające warunek")

    frame_funcs = {
        "Volume": app.volume,
        "Short-Time Energy": app.STE,
        "Zero Crossing Rate": app.ZCR,
        "Autocorrelation": app.autocorrelation,
    }
    frame_level_func_name = st.selectbox(
        "Wybierz cechę na poziomie ramki:", list(frame_funcs.keys())
    )
    frame_level_func = frame_funcs[frame_level_func_name]
    frames, func_values, upper_bound = app.get_frame_level_func_range(
        frame_level_func, frame_duration
    )

    min_val, max_val = st.slider(
        key="slider_page_2",
        label="Wybierz zakres wartości:",
        min_value=0.0,
        max_value=upper_bound,
        value=(0.0, upper_bound),
        step=upper_bound / 50,
    )
    try:
        samples = app.display_frames_signal_between_values(
            frames, func_values, min_val, max_val
        )
        st.audio(samples, format="audio/wav", sample_rate=app.frame_rate)
    except:
        st.write("Brak ramek pomiędzy zadanymi poziomami :confused:")
