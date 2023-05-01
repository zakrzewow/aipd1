import streamlit as st
import tabs.sidebar

import tabs.project_1.przebieg_czasowy
import tabs.project_1.cechy_poziom_ramki
import tabs.project_1.detekcja_ciszy
import tabs.project_1.fragmenty_dzwieczne
import tabs.project_1.muzyka_mowa
import tabs.project_1.cechy_poziom_klipu
import tabs.project_1.pobieranie_markerow

import tabs.project_2.frequency_features
import tabs.project_2.time_frequency_domain_plots
import tabs.project_2.spectrogram
import tabs.project_2.laryngeal_frequency


def main():
    ## page config
    st.set_page_config(page_title="AiPD projekt 1 & 2", layout="wide")

    ## tabs
    TABS = {
        "Przebieg czasowy pliku audio": tabs.project_1.przebieg_czasowy,
        "2️⃣ Parametry dźwieku z dziedziny częstotliwości": tabs.project_2.frequency_features,
        "2️⃣ Wykresy sygnału w dziedzinie czasu i częstotliwości": tabs.project_2.time_frequency_domain_plots,
        "2️⃣ Spektogram": tabs.project_2.spectrogram,
        "2️⃣ Częstotliwość kratniowa": tabs.project_2.laryngeal_frequency,
        "1️⃣ Cechy na poziomie ramki": tabs.project_1.cechy_poziom_ramki,
        "1️⃣ Detekcja ciszy": tabs.project_1.detekcja_ciszy,
        "1️⃣ Określanie fragmentów dźwięcznych i bezdźwięcznych": tabs.project_1.fragmenty_dzwieczne,
        "1️⃣ Określanie fragmentów muzyka vs. mowa": tabs.project_1.muzyka_mowa,
        "1️⃣ Analiza na poziomie klipu": tabs.project_1.cechy_poziom_klipu,
        "1️⃣ Pobieranie markerów określających granice": tabs.project_1.pobieranie_markerow,
    }

    ## sidebar
    selected_tab, file_name, app, frequencyApp = tabs.sidebar.run(TABS)

    tab_module = TABS.get(selected_tab, None)
    if tab_module:
        tab_module.run(selected_tab, file_name, app, frequencyApp)

    if selected_tab == "Informacje":
        st.write(
            "Ta aplikacja umożliwia wybór pliku *.wav (z przykładów lub wgranie własnego), przegląd metryk oraz przeprowadzenie analizy."
        )


if __name__ == "__main__":
    main()
