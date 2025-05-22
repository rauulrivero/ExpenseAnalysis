import streamlit as st
import json

class StreamlitApp:
    def __init__(self, api_handler):
        self.api_handler = api_handler

    def run(self):
       