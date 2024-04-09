import streamlit as st
from io import StringIO
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import os

client_id = st.secrets['AZURE_CLIENT_ID']
tenant_id = st.secrets['AZURE_TENANT_ID']
client_secret = st.secrets['AZURE_CLIENT_SECRET']
account_url = st.secrets["AZURE_STORAGE_URL"]

# create a credential 
credentials = ClientSecretCredential(
    client_id = client_id, 
    client_secret= client_secret,
    tenant_id= tenant_id
)

def get_supp_set_list():
    support_set_folder = "./support_sets"
    return os.listdir(support_set_folder) 

def list_supp_sets():
    support_set_folder = "./support_sets"
    count = 1
    supp_set_list = get_supp_set_list()
    st.header("Support Sets")
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
