import streamlit as st


class Session:
    def __init__(self):
        pass

    def __getitem__(self, key):
        if key not in st.session_state:
            return None
        return st.session_state[key]

    def __setitem__(self, key, value):
        st.session_state[key] = value

    def contains(self, key):
        if key not in st.session_state:
            return False
        return True

    def remove(self, key):
        if key not in st.session_state:
            return False
        del st.session_state[key]
        return True
