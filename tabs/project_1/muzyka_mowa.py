import streamlit as st
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def run(selected_tab: str, file_name: str, app: TimeApp, frequencyApp: FrequencyApp):
    st.header("Określanie fragmentów muzyka vs. mowa")

    st.write(
        "Ramka jest uznawana za muzykę \
                jeśli wartości LSTER oraz ZSRD znajdują się w żądanych przedziałach"
    )

    lster_list = app.proper_LSTER()

    if len(lster_list) == 1:
        st.write("Audio jest zbyt krótkie by analizować pod tym kątem")
    else:
        fig = app.plot_proper_music_metric(lster_list, "LSTER")
        st.plotly_chart(fig)

        col_vol_min, col_vol_max = st.columns(2)
        lster_threshord = col_vol_min.number_input(
            "LSTER wartość graniczna", 0.0, 2.0, 0.02, 0.005
        )

        fig = app.plot_music_vs_talks(lster_list, lster_threshord, "Muzyka vs. mowa")
        st.plotly_chart(fig)
