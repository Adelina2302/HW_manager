import streamlit as st
from HWs import HW1, HW2

# Page config
st.set_page_config(page_title="HW Manager", page_icon="🗂️", layout="wide")
st.title("🗂️ HW Manager")

st.write("Select the homework below:")

# Radio buttons to switch pages, default is HW2
page = st.radio(
    "Choose HW:",
    ("HW1", "HW2"),
    index=1  # HW2 opens by default
)

# Display the selected page
if page == "HW1":
    HW1.app()
else:
    HW2.app()
