"""YMC 4.8 TEST: safe Streamlit AI status panel, without chart/layout changes."""
from html import escape
from ai_bridge import get_ai_status

def build_ai_panel(df):
    rows = []
    for horizon in (10, 15):
        result = get_ai_status(df, horizon)
        status = escape(str(result['status']))
        detail = escape(str(result['detail']))
        signal = escape(str(result['signal']))
        rows.append(f'<div class="ymc-ai-status"><strong>IA {horizon}M: {status}</strong><span> {signal} · {detail}</span></div>')
    return '<div class="ymc-ai-panel">' + ''.join(rows) + '</div>'

def render_ai_panel(df):
    import streamlit as st
    st.markdown(build_ai_panel(df), unsafe_allow_html=True)
