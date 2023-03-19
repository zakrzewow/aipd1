import streamlit as st
import pandas as pd
import numpy as np
import os

from IPython.display import clear_output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as pltz
import datetime
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from ipywidgets import widgets
from IPython.display import clear_output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as pltz
import plotly.io as pio
pio.renderers.default = 'notebook'
from pydub import AudioSegment
import plotly.express as px
import numpy as np
from modules.app import App, Frame
import plotly.express as px
import librosa


# Define the function for analysis
def analyze_data(data):
    if data is not None:
        # Perform analysis on the data here
        return "Analysis complete!"

# Define the Streamlit app
def main():
    # Set up the menu with file selector
    st.set_page_config(page_title="Data Analysis", layout="wide")
    directory = "./samples/Nagrania_01/2_02/Znormalizowane/"
    file_list = os.listdir(directory)
    file_name = st.sidebar.selectbox("Select a file", file_list)

    # Define the tabs
    tabs = ["Analyze Data on frame level", "Analyze Data on clip level", 
            "Download parameters", "About"]
    clicked_tab = st.sidebar.radio("Select a tab", tabs)
    file_path = os.path.join(directory, file_name)
    app = App(file_path)

    if clicked_tab == "Analyze Data on frame level":

        file_path = os.path.join(directory, file_name)
        clicked_tab = st.radio("Analiza", ["Cisza", "Fragmenty bezdźwięczne", "Muzyka"])

        frame_duration = st.slider("Select frame duration (in milliseconds)", min_value=1, max_value=25, value=5)
        

        col1, col2 = st.columns(2)
        with col1:
            col11, col12 = st.columns(2)
            with col11:
                min_val_volume = st.number_input(key="Minimum Value Volume", label="Min", min_value=0, max_value = 10**4, value = 0, step= 1)
            with col12:
                max_val_volume = st.number_input(key="Maximum Value Volume", label="Max", min_value=0, max_value= 10**4, value = 0, step= 1)
            
            plot_volume = app.plot_frame_level_feature(app.volume, "VOLUME", frame_duration_miliseconds = frame_duration,
                                            min_val =  min_val_volume, max_val= max_val_volume) 
            st.plotly_chart(plot_volume)

            col13, col14 = st.columns(2)
            with col13:
                min_val_ste = st.number_input(label="Min", min_value=0, max_value = 10**4, value = 0, step= 1)
            with col14:
                max_val_ste= st.number_input(label="Max", min_value=0, max_value= 10**4, value = 0, step= 1)
            
            plot_ste = app.plot_frame_level_feature(app.STE, "STE", frame_duration_miliseconds = frame_duration,
                                            min_val =  min_val_ste, max_val= max_val_ste) 
            st.plotly_chart(plot_ste)


            col15, col16 = st.columns(2)
            with col15:
                min_val_f0_cor = st.number_input(label="Min", min_value=-5, max_value = 10**4, value = 0, step= 1)
            with col16:
                max_val_f0_cor = st.number_input(label="Max", min_value=-5, max_value= 10**4, value = 0, step= 1)
            
            plot_f0_cor = app.plot_frame_level_feature(app.F0_Cor, "F0_Cor", frame_duration_miliseconds = frame_duration,
                                            min_val =  min_val_f0_cor, max_val= max_val_f0_cor) 
            st.plotly_chart(plot_f0_cor)


            with col2:
                col21, col22 = st.columns(2)
                with col21:
                    min_val_zcr= st.number_input(label="Min", min_value=-1, max_value = 10**4, value = 0, step= 1)
                with col22:
                    max_val_zcr= st.number_input(label="Max", min_value=-1, max_value= 10**4, value = 0, step= 1)
                
                plot_zcr= app.plot_frame_level_feature(app.ZCR, "ZCR", frame_duration_miliseconds = frame_duration,
                                                min_val =  min_val_zcr, max_val= max_val_zcr) 
                st.plotly_chart(plot_zcr)


                col23, col24 = st.columns(2)
                with col23:
                    min_val_st= st.number_input(label="Min", min_value=-2, max_value = 10**4, value = 0, step= 1)
                with col24:
                    max_val_st= st.number_input(label="Max", min_value=-3, max_value= 10**4, value = 0, step= 1)
                
                plot_sr = app.plot_frame_level_feature(app.SR, "SR", frame_duration_miliseconds = frame_duration,
                                                min_val =  min_val_st, max_val= max_val_st) 
                st.plotly_chart(plot_sr)


                col25, col26 = st.columns(2)
                with col25:
                    min_val_f0_amdf = st.number_input(label="Min", min_value=-6, max_value = 10**4, value = 0, step= 1)
                with col26:
                    max_val_f0_amdf= st.number_input(label="Max", min_value=-6, max_value= 10**4, value = 0, step= 1)
                
                plot_f0_amdf = app.plot_frame_level_feature(app.F0_AMDF, "F0_Cor", frame_duration_miliseconds = frame_duration,
                                                min_val =  min_val_f0_amdf, max_val= max_val_f0_amdf) 
                st.plotly_chart(plot_f0_amdf)



            
                    
        # Define a dropdown to allow the user to select the frame-level function
        frame_funcs = {"Volume": app.volume,
                       'Short-Time Energy': app.STE,
                         "Zero Crossing Rate": app.ZCR,
                         "Silten ratio": app.SR}
        frame_level_func_name = st.selectbox("Frame-Level Function", list(frame_funcs.keys()))
        frame_level_func = frame_funcs[frame_level_func_name]

        # Display the frames between the minimum and maximum values
        min_val, max_val = st.slider("Select a range of values", 0, 32767, (5000, 10000))
        try:
            audio_data = app.display_frames_between_values(frame_level_func, min_val, max_val)
            st.audio(audio_data, format='audio/wav') 
        except:
            st.write("No frames found between the specified values.")

    if clicked_tab == "Analyze Data on clip level":
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


        print(df)
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




    if clicked_tab == "Download parameters":

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



        frame_funcs = {"Volume": app.volume,
                       'Short-Time Energy': app.STE,
                         "Zero Crossing Rate": app.ZCR}
        frame_level_func_name = st.selectbox("Frame-Level Function", list(frame_funcs.keys()))
        frame_level_func = frame_funcs[frame_level_func_name]

        frame_duration = st.slider("Select frame duration (in milliseconds)", min_value=1, max_value=25, value=5)
        frames = [frame for frame in app.frame_generator(frame_duration)]

        y = [frame_level_func(frame) for frame in frames]
        # Add a button to save the frame-level function values to a CSV file
        if st.button("Save to CSV"):
            audio_name = app.filepath.split("/")[-1].split(".")[0]
            filename = f"{audio_name}_{frame_level_func_name}_{frame_duration}ms.csv"
            df = pd.DataFrame({"Frame": range(len(y)), frame_level_func_name: y})
            df.to_csv(filename, index=False)
            st.success(f"File saved to {filename}")


    if clicked_tab == "About":
        # Display information about the app
        st.write("This app allows you to select a file from a directory, view the data, and perform analysis on it.")
    
    
    


# Run the app
if __name__ == "__main__":
    main()