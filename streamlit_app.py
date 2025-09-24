import streamlit as st

pg = st.navigation(
    {
        "HWs": [
            st.Page("HWs/HW4.py", title="HW4"), 
            st.Page("HWs/HW3.py", title="HW3"), 
            st.Page("HWs/HW2.py", title="HW2"),
            st.Page("HWs/HW1.py", title="HW1"),
        ]
    }
)

pg.run()