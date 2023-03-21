import os
import streamlit as st
import numpy as np
import pandas as pd
from modules.app import App, Frame


def main():
    ## page config
    st.set_page_config(page_title="AiPD projekt 1", layout="wide")

    ## sidebar

    # select file
    directory = "./samples/Nagrania_01/2_02/Znormalizowane/"
    file_list = os.listdir(directory)
    st.sidebar.header("Wybierz plik")
    file_name = st.sidebar.selectbox("file_selectbox", file_list, label_visibility="collapsed")

    file_path = os.path.join(directory, file_name)
    app = App(file_path)

    # audio
    st.sidebar.audio(app.samples, sample_rate=app.frame_rate)

    # tabs
    tabs = [
        "Przebieg czasowy pliku audio",
        "Cechy na poziomie ramki",
        "Detekcja ciszy",
        "Analiza na poziomie klipu", 
        "Pobieranie parametrów", 
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

        frame_duration = st.slider("Długość ramki (w milisekundach):", min_value=10, max_value=40, value=25, step=5)
        
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
            "Silent ratio": app.SR
        }
        frame_level_func_name = st.selectbox("Wybierz cechę na poziomie ramki:", list(frame_funcs.keys()))
        frame_level_func = frame_funcs[frame_level_func_name]
        frames, func_values, upper_bound = app.get_frame_level_func_range(frame_level_func, frame_duration)

        (min_val), max_val = st.slider("Wybierz zakres wartości:", 0.0, upper_bound, (0.0, upper_bound), upper_bound / 50)
        try:
            samples = app.display_frames_signal_between_values(frames, func_values, min_val, max_val)
            st.audio(samples, format='audio/wav', sample_rate=app.frame_rate) 
        except:
            st.write("Brak ramek pomiędzy zadanymi poziomami :confused:")


    if selected_tab == "Detekcja ciszy":
        st.header("Detekcja ciszy")

        st.subheader("Silent Ratio - ustawienia")
        st.write("Ramka jest uznawana za ciszę jeśli wartości Volume oraz ZCR znajdują się w żądanym przedziale.")

        frame_duration = st.slider("Długość ramki (w milisekundach):", min_value=10, max_value=40, value=25, step=5)

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
                fig_layout_kwargs=dict(width=400, height=300)
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
                fig_layout_kwargs=dict(width=400, height=300)
            ) 
            st.plotly_chart(plot_zcr)

        st.subheader("Zaznaczenie ciszy na wykresie")
        plot_sample_with_sr = app.plot_sample_with_SR(frame_duration, vol_min_val, vol_max_val, zcr_min_val, zcr_max_val) 
        st.plotly_chart(plot_sample_with_sr)

    if selected_tab == "Analiza na poziomie klipu":
        # nie działa
        
        # Create empty dataframe
        df = pd.DataFrame(columns=['Filename', 'VSTD'])

        # Function to calculate features and add row to dataframe
        def process_file(file_path, file_name):
            app = App(file_path)
            vstds = []
            for i, frame in enumerate(app.frames):
                vstds.append(app.VSTD([frame]))
            row = {'Filename': file_name,  'VSTD': sum(vstds)/len(vstds)}
            df.loc[len(df)+1] = row

        # Streamlit apps
        st.title('Audio Feature Analysis')
        file_name = st.selectbox("Select a file 2", file_list)
        file_path = os.path.join(directory, file_name)
        if file_name is not None:
            process_file(file_path, file_name)
            st.write('### Results')
            st.write(df)
            st.write('### Scatter Plots')
            st.write('TODO: Add scatter plots for LSTER and VSTD')


    if selected_tab == "Pobieranie parametrów":

        # Define the column names for the table
        col1 = "Frame-Level Function"
        col2 = "Parameter"

        # Define the default values for the table
        data = pd.DataFrame({
            col1: ["Volume", "ZCR", "STE"],
            col2: [0.5, 0.1, 0.2]
        })

        # Create the table
        table = st.experimental_data_editor(data, num_rows="dynamic")

        # Define a button to export the table to Excel
        if st.button("Export to Excel"):
            data.to_excel("table.xlsx", index=False)


    if selected_tab == "Informacje":
        st.write("This app allows you to select a file from a directory, view the data, and perform analysis on it.")


if __name__ == "__main__":
    main()
