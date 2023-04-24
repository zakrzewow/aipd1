import os
import streamlit as st
import pandas as pd
from modules.timeApp import TimeApp
from modules.frequencyApp import FrequencyApp


def main():
    ## page config
    st.set_page_config(page_title="AiPD projekt 1", layout="wide")

    ## sidebar

    # select file
    st.sidebar.subheader("Wgraj własny plik .wav")
    uploaded_file = st.sidebar.file_uploader("file_selectbox", "wav", label_visibility="collapsed")
    
    st.sidebar.header("LUB")
    st.sidebar.subheader("Wybierz głos")
    users = ["Kacper", "Grzegorz", "głos kobiecy"]
    user = st.sidebar.selectbox("user_selectbox", users, label_visibility="collapsed", disabled=uploaded_file is not None)

    directory = f"./samples/{user}/Znormalizowane/"
    file_list = os.listdir(directory)
    st.sidebar.subheader("Wybierz plik")
    file_name = st.sidebar.selectbox("file_selectbox", file_list,
                                    label_visibility="collapsed",
                                    disabled=uploaded_file is not None)

    if uploaded_file is None:
        file_path = os.path.join(directory, file_name)
        app = TimeApp(file_path)
    else:
        app = TimeApp(uploaded_file)

    # audio
    st.sidebar.audio(app.samples, sample_rate=app.frame_rate)

    # tabs
    tabs = [
        "Przebieg czasowy pliku audio",
        "Cechy na poziomie ramki",
        "Detekcja ciszy",
        "Określanie fragmentów dźwięcznych i \n bezdźwięcznych",
        "Określanie fragmentów muzyka vs. \n mowa",
        "Analiza na poziomie klipu", 
        "Pobieranie markerów określających granice", 
        "Parametry dźwieku z dziedziny częstotliwości",
        "Wykresy sygnału w dziedzinie czasu i częstotliwości",
        "Spektogram",
        "Częstotliwość kratniowa",
        "Informacje"
    ]
    selected_tab = st.sidebar.radio("tab_radio", tabs, label_visibility="collapsed")

    ## tabs
    
    if selected_tab == "Przebieg czasowy pliku audio":
        st.header("Przebieg czasowy pliku audio")
        plot_sample = app.plot_sample() 
        st.plotly_chart(plot_sample)


    if selected_tab == "Cechy na poziomie ramki":
        st.header("Cechy sygnału audio na poziomie ramki")

        frame_duration = st.slider(key = "slider_page_1",
                                    label = "Długość ramki (w milisekundach):", 
                                    min_value=10, 
                                    max_value=40, 
                                    value=25, 
                                    step=5)
        
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

        #autocorrealation
        frame_numbers_input = st.text_input("Enter frame numbers to plot separated by commas (e.g. 1,2,3):", "1,2,3")
        frame_numbers = [int(x.strip()) for x in frame_numbers_input.split(",")]
        plot_autocorr =  app.plot_autocorrelation(
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
        frame_level_data = app.get_frame_level_export_data(frame_duration_miliseconds=frame_duration)        
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
            "Autocorrelation": app.autocorrelation
        }
        frame_level_func_name = st.selectbox("Wybierz cechę na poziomie ramki:", list(frame_funcs.keys()))
        frame_level_func = frame_funcs[frame_level_func_name]
        frames, func_values, upper_bound = app.get_frame_level_func_range(frame_level_func, frame_duration)

        min_val, max_val = st.slider(key = "slider_page_2",
                                        label =   "Wybierz zakres wartości:",
                                        min_value = 0.0, max_value = upper_bound,
                                        value = (0.0, upper_bound),
                                        step = upper_bound / 50)
        try:
            samples = app.display_frames_signal_between_values(frames, func_values, min_val, max_val)
            st.audio(samples, format='audio/wav', sample_rate=app.frame_rate) 
        except:
            st.write("Brak ramek pomiędzy zadanymi poziomami :confused:")


    if selected_tab == "Detekcja ciszy":
        st.header("Detekcja ciszy")

        st.subheader("Silent Ratio - ustawienia")
        st.write("Ramka jest uznawana za ciszę jeśli wartości Volume oraz ZCR znajdują się w żądanym przedziale.")

        frame_duration = st.slider(key = "slider_page_3", label = "Długość ramki (w milisekundach):", min_value=10, max_value=40, value=25, step=5)

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
                fig_layout_kwargs=dict(width=500, height=300)
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
                fig_layout_kwargs=dict(width=500, height=300)
            ) 
            st.plotly_chart(plot_zcr)

        st.subheader("Zaznaczenie ciszy na wykresie")
        plot_sample_with_sr = app.plot_sample_with_SR(frame_duration, vol_min_val, vol_max_val, zcr_min_val, zcr_max_val) 
        st.plotly_chart(plot_sample_with_sr)

    
    if selected_tab == "Określanie fragmentów dźwięcznych i \n bezdźwięcznych":
        st.header("Określanie fragmentów dźwiecznych i bezdźwięcznych")

        st.subheader("Fragment bezdźwięczny - ustawienia")
        st.write("Ramka jest uznawana za fragment bezdźwięczny \
                  jeśli wartości STE oraz ZCR znajdują się w żądanych przedziałach")

        frame_duration = st.slider(key = "slider_page_5",
                                    label = "Długość ramki (w milisekundach):", 
                                    min_value=5, 
                                    max_value=40, 
                                    value=25,
                                    step=5)

        col_vol, col_zcr = st.columns(2)
        with col_vol:
            st.markdown("#### STE")
            col_vol_min, col_vol_max = st.columns(2)
            ste_min_val = col_vol_min.number_input(key = "ste_min_np_page4",
                                                    label = "STE MIN",
                                                    min_value = 0.00,
                                                    max_value = 2.00,
                                                    value = 0.000,
                                                    step = 0.005, 
                                                    format = "%.3f")
            
            ste_max_val = col_vol_max.number_input(key = "ste_max_np_page4",
                                                    label = " STE MAX",
                                                    min_value = 0.00,
                                                    max_value = 2.00,
                                                    value = 0.01,
                                                    step = 0.005,  
                                                    format = "%.3f")
            # volume
            plot_volume = app.plot_frame_level_feature(
                frame_level_func=app.STE, 
                plot_title="", 
                frame_duration_miliseconds=frame_duration,
                min_val=ste_min_val,
                max_val=ste_max_val,
                fig_layout_kwargs=dict(width=500, height=300)
            ) 
            st.plotly_chart(plot_volume)

        with col_zcr:
            st.markdown("#### ZCR")
            col_zcr_min, col_zcr_max = st.columns(2)
            zcr_min_val = col_zcr_min.number_input(key = "zcr_min_np_page_1",
                                                    label = "ZCR MIN",
                                                    min_value= 0.0,
                                                    max_value= 2.0,
                                                    value = 0.4,
                                                    step = 0.05)
            
            zcr_max_val = col_zcr_max.number_input(key = "zcr_max_np_page_2",
                                                    label = "ZCR MAX",
                                                    min_value = 0.0,
                                                    max_value = 2.0,
                                                    value = 1.0,
                                                    step = 0.05)
            # zcr
            plot_zcr = app.plot_frame_level_feature(
                frame_level_func=app.ZCR, 
                plot_title="", 
                frame_duration_miliseconds=frame_duration,
                min_val=zcr_min_val,
                max_val=zcr_max_val,
                fig_layout_kwargs=dict(width=500, height=300)
            ) 
            st.plotly_chart(plot_zcr)

        st.subheader("Zaznaczenie fragmentów bezdźwięcznych na wykresie")
        plot_sample_with_sr = app.plot_sample_with_SR(frame_duration, ste_min_val, ste_max_val, zcr_min_val, zcr_max_val) 
        st.plotly_chart(plot_sample_with_sr)



    if selected_tab == "Określanie fragmentów muzyka vs. \n mowa":
        st.header("Określanie fragmentów muzyka vs. mowa")

        st.write("Ramka jest uznawana za muzykę \
                  jeśli wartości LSTER oraz ZSRD znajdują się w żądanych przedziałach")


        lster_list = app.proper_LSTER()

        if len(lster_list) == 1:
            st.write("Audio jest zbyt krótkie by analizować pod tym kątem")
        else:
            fig = app.plot_proper_music_metric(lster_list, "LSTER")
            st.plotly_chart(fig)
            
            col_vol_min, col_vol_max = st.columns(2)
            lster_threshord = col_vol_min.number_input("LSTER wartość graniczna", 0.0, 2.0, 0.02, 0.005)
            
            fig = app.plot_music_vs_talks(lster_list,
                                        lster_threshord,
                                        "Muzyka vs. mowa")
            st.plotly_chart(fig)


    if selected_tab == "Analiza na poziomie klipu":
        st.header("Analiza na poziomie klipu")

        # vstd = app.VSTD(app.frames)
        # vdr = app.VDR(app.frames)
        # vu = app.VU(app.frames)
        # lster = app.LSTER(app.frames)
        # energy_entropy =  app.energy_entropy(app.frames)
        # hzcrr = app.HZCRR(app.frames)
        # spectral_centroid  = app.spectral_centroid_std(app.frames)
        # spectral_bandwidth  = app.spectral_bandwidth_std(app.frames)

        # data = {'Values': [vstd, vdr, vu, 
        #                    lster, energy_entropy, hzcrr,
        #                    spectral_centroid, spectral_bandwidth]}

        # # Create DataFrame
        # df = pd.DataFrame(data, index=['VSTD',
        #                                'VDR',
        #                                'VU',
        #                                'LSTER',
        #                                'ENERGY ENTROPY',
        #                                'HZCRR',
        #                                'SPECTRAL CENTROID',
        #                                'SPECTRAL BANDWIDTH', 
        # ])

        # # Select only one column
        # df_column = df[['Values']]

        # # Rename index
        # df_column.index.name = 'Metrics'

        # # Display table in Streamlit
        # st.write(df_column)


        # # Plot the standard deviation of spectral contrast values
        # spectral_contrast_std_values = app.spectral_contrast_std(app.frames)
        # if len(spectral_contrast_std_values) == 0:
        #     st.write("Audio jest na za którkie na analizę Spectral Contrast Std")
        # else:
        #     plot_spectral_contrast_std = app.plot_spectral_contrast_std(spectral_contrast_std_values)
        #     st.plotly_chart(plot_spectral_contrast_std)

        # # Plot spectrogram
        # st.subheader("Spectrogram")
        # spectrogram = app.plot_spectrogram()
        # st.plotly_chart(spectrogram)

        # ## Download data
        # st.subheader("Zapis parametrów")
        # clip_level_data = app.get_clip_level_export_data()        
        # st.download_button(
        #     label="Pobierz parametry (.csv)",
        #     data=clip_level_data.to_csv(),
        #     file_name=f"{file_name.split('.')[0]}_frame_10.csv",
        #     mime="text/csv",
        # )


    if selected_tab == "Pobieranie markerów określających granice":
        st.header("Analiza na poziomie klipu")

        # Define the column names for the table
        col1 = "Frame-Level Function"
        col2 = "Parameter"

        # Define the default values for the table
        frame_level_functions = [
            "Volume", "Short-Time Energy", "Zero Crossing Rate", "Silent ratio"
        ]
        clip_level_functions = [
            "VSTD", "VDR", "VU", "LSTER", "ENERGY ENTROPY", "HZCRR"
        ]
        all_functions = frame_level_functions + clip_level_functions

        data = pd.DataFrame({
            col1: all_functions,
            col2: [0.5] * len(all_functions)  # Change this list to set default parameter values
        })

        # Create the table
        table = st.experimental_data_editor(data, num_rows="dynamic")

        st.download_button(
            label="Pobierz (.csv)",
            data=table.to_csv(),
            file_name=f"table.csv",
            mime="text/csv",
        )

    if uploaded_file is None:
        file_path = os.path.join(directory, file_name)
        frequencyApp = FrequencyApp(file_path)
    else:
        frequencyApp = FrequencyApp(uploaded_file)

    if selected_tab == "Parametry dźwieku z dziedziny częstotliwości":
        st.header("Parametry dźwieku z dziedziny częstotliwości")
        
        frame_duration_frec = st.slider(key = "slider_frec",
                                    label = "Długość ramki (w milisekundach):", 
                                    min_value=10, 
                                    max_value=40, 
                                    value=25, 
                                    step=5)
        
        # volume
        plot_volume_frec = frequencyApp.plot_frame_level_feature(
            frame_level_func=frequencyApp.volume, 
            plot_title="VOLUME", 
            frame_duration_miliseconds=frame_duration_frec,
        ) 
        st.plotly_chart(plot_volume_frec)

    if selected_tab == "Wykresy sygnału w dziedzinie czasu i częstotliwości":
        st.header("Wykresy sygnału w dziedzinie czasu i częstotliwości")

        available_window_names = ["okno prostokątne", "okno trójkątne", "boxcar", "hann", "hamming", "blackman", "bartlett"]
        selected_window_name = st.radio(
            label="Wybierz funkcję okienkową:",
            options=available_window_names,
            index=0,  # Default selected option
        )


        time_domain_spectrum_plot = frequencyApp.plot_time_domain_spectrum_windows(window_name=selected_window_name)
        st.plotly_chart(time_domain_spectrum_plot)

        spectrum_plot = frequencyApp.plot_spectrum_windows(window_name = selected_window_name)
        st.plotly_chart(spectrum_plot)

    if selected_tab == "Spektogram":
        st.header("Spektogram")
        # Create a window type selector
        window_options =  ["okno prostokątne", "okno trójkątne", "boxcar", "hann", "hamming", "blackman", "bartlett"]
        selected_window = st.selectbox("Wybierz funkcję okienkową", window_options)
        print(window_options)

        # Create sliders for NFFT and noverlap
        NFFT = st.slider("Długość ramki", 128, 4096, 1024, step=128)
        noverlap = st.slider("Długość nakładania ramek", 0, NFFT - 128, NFFT // 2, step=128)

        # Call the create_spectrogram and plot_spectrogram methods
        frequencyApp.create_spectrogram(NFFT, selected_window, noverlap)
        spectrogram_plot = frequencyApp.plot_spectrogram(NFFT, selected_window, noverlap)

        # Display the spectrogram
        st.plotly_chart(spectrogram_plot)


    if selected_tab == "Częstotliwość kratniowa":
        st.header("Częstotliwośc krtaniowa")
        st.subheader("Cepstrum")
        cepstrum_signal_plot = frequencyApp.visualize_cepstrum_signal()
        st.plotly_chart(cepstrum_signal_plot)
        st.subheader("Czestotliwość kratniowa (F0)")
        window_size = st.slider("Rozmiar okna", min_value=256, max_value=2**13, value=256, step=128)
        step_size = st.slider("Długośc nakładania ramek", min_value=256, max_value=2**13, value=256, step=128)
        
        window_options = ["okno prostokątne", "okno trójkątne", "boxcar", "hann", "hamming", "blackman", "bartlett"]
        selected_window_function = st.selectbox("Wybierz funkcję okienkową", window_options)

        # Use the selected window function when calling visualize_laryngeal_frequency
        f0_signal_plot = frequencyApp.visualize_laryngeal_frequency(window_size, step_size, selected_window_function)
        st.plotly_chart(f0_signal_plot)
        
        st.subheader("Cześctotliość tonu podstawowego dla całego klipu wynosi:")
        st.markdown(f"**{frequencyApp.laryngeal_frequency(window_function = selected_window_function)[1][0]}Hz**")

    if selected_tab == "Informacje":
        st.write("Ta aplikacja umożliwia wybór pliku *.wav (z przykładów lub wgranie własnego), przegląd metryk oraz przeprowadzenie analizy.")


if __name__ == "__main__":
    main()
