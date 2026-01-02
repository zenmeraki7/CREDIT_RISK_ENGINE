import streamlit as st
import plotly.graph_objects as go

@st.cache_resource
def gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={'axis': {'range': [0, 100]}}
    ))
    fig.update_layout(height=300, transition_duration=0)
    return fig
