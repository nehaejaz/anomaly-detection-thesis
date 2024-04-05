import streamlit as st
from io import StringIO
import os

def get_supp_set_list():
    support_set_folder = "./support_sets"
    return os.listdir(support_set_folder)

def list_supp_sets():
    support_set_folder = "./support_sets"
    count = 1
    supp_set_list = get_supp_set_list()
    for index, supp_set_name in enumerate(supp_set_list):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"{count}. {supp_set_name}")
        with col2:
            if st.button("Delete", type="primary", key=count):
                #delete that set
                file_path = os.path.join(support_set_folder, supp_set_name)
                os.remove(file_path)
                # Refresh the list
                supp_set_list = get_supp_set_list()
                st.experimental_rerun()  # Rerun the script to update the UI
        count +=1

list_supp_sets()