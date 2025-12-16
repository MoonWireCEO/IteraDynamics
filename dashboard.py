# src/dashboard.py
# 🦅 ARGUS MISSION CONTROL - V6.2 (FINAL VERBOSE LOG FIX)
# Reads the verbose argus_execution.log file.

import streamlit as st
import pandas as pd
import json
import re
import sys
import os
import time
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path

# --- CONNECT TO REAL BROKER ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.real_broker import RealBroker
    BROKER_AVAILABLE = True
except Exception as e:
    BROKER_AVAILABLE = False
    BROKER_ERROR = e

# --- CONFIGURATION ---
CHART_FILE = Path("data/flight_recorder.csv") # Used for the performance curve
EXECUTION_LOG = Path("data/argus_execution.log") # Used for the terminal log display

st.set_page_config(page_title="Argus Mission Control", page_icon="🦅", layout="wide")

# --- CUSTOM CSS (FINAL CROSS-BROWSER FIX) ---
st.markdown("""
    <style>
    /* 1. Main Background */
    .stApp { background-color: #0E1117; }
    
    /* 2. Metric Styling */
    [data-testid="stMetricLabel"] {
        font-size: 16px !important; color: #FFFFFF !important; font-weight: 700 !important; opacity: 1 !important;
    }
    [data-testid="stMetricValue"] {
        font-family: "Source Code Pro", monospace; font-size: 32px !important; color: #00FF88 !important;
    }
    
    /* 3. Headers */
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 800 !important; }
    
    /* 4. Chart Transparency */
    .js-plotly-plot .plotly .main-svg { background: rgba(0,0,0,0) !important; }
    
    /* 5. FINAL FIX: Target the container of the code block for cross-browser consistency */
    div.stCodeBlock {
        background-color: #1a1a1a !important; /* Dark background for the outer container */
        border: 1px solid #333 !important;
        border-radius: 5px !important;
        max-height: 400px;
        overflow-y: scroll;
        padding: 15px; /* Add padding inside the container */
    }

    /* 6. Target the text inside the code block */
    code, pre {
        font-family: 'Source Code Pro', monospace !important;
        font-size: 12px !important;
        color: #cccccc !important; /* Light grey text */
        background-color: transparent !important; /* Crucial: make the text background transparent */
        border: none !important; /* Remove internal border */
    }
    </style>
    """, unsafe_allow_html=True)

# --- LIVE DATA ENGINE ---
def get_live_data():
    """Connects to RealBroker to get ACTUAL wallet balance."""
    if not BROKER_AVAILABLE:
        return 0.0, 0.0, f"Broker Import Failed: {BROKER_ERROR}"
    
    try:
        broker = RealBroker()
        cash = broker.cash
        btc_qty = broker.positions
        return cash, btc_qty, "CONNECTED"
    except Exception as e:
        return 0.0, 0.0, f"Connection Error: {e}"

def get_live_ticker():
    """Fetches the REAL-TIME price of BTC from Coinbase."""
    if not BROKER_AVAILABLE: return 0.0
    try:
        broker = RealBroker()
        product = broker.client.get_product("BTC-USD")
        
        if hasattr(product, 'price'):
            return float(product.price)
        elif isinstance(product, dict):
            return float(product.get('price', 0))
        return 0.0
    except:
        return 87150.00 

def scrape_equity_curve():
    """Builds chart from CSV logs."""
    if not CHART_FILE.exists(): return None
    try:
        df = pd.read_csv(CHART_FILE)
        if 'Timestamp' in df.columns and 'Equity' in df.columns:
            df['dt'] = pd.to_datetime(df['Timestamp'])
            if 'Equity' not in df.columns: return None 
            return df
    except: return None
    return None

# --- VISUALIZATION HELPERS (Unchanged) ---
def create_equity_chart(df):
    fig = go.Figure()
    if df is not None and not df.empty:
        fig.add_trace(go.Scatter(x=df['dt'], y=df['Equity'], mode='lines', name='Equity',
            line=dict(color='#00FF88', width=3), fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.05)'))
        min_y = df['Equity'].min(); max_y = df['Equity'].max()
        buffer = (max_y - min_y) * 0.5 if max_y != min_y else 50
        fig.update_layout(yaxis=dict(range=[min_y - buffer, max_y + buffer]))
    
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color="white"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='#333333'),
        xaxis=dict(showgrid=False, linecolor='#333333'))
    return fig

def make_gauge(confidence):
    fig = go.Figure(go.Indicator(mode = "gauge+number", value = confidence * 100,
        number = {'suffix': "%", 'font': {'color': "white", 'size': 24}},
        title = {'text': "AI CONVICTION", 'font': {'size': 16, 'color': "white"}},
        gauge = {'axis': {'range': [0, 100], 'tickcolor': "white"}, 'bar': {'color': "#636EFA"},
            'bgcolor': "#0E1117", 'borderwidth': 2, 'bordercolor': "#333",
            'steps': [{'range': [0, 50], 'color': "rgba(239, 85, 59, 0.3)"},
                      {'range': [50, 70], 'color': "rgba(255, 205, 86, 0.3)"},
                      {'range': [70, 100], 'color': "rgba(0, 204, 150, 0.3)"}],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 70}}))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# --- MAIN DASHBOARD ---
st.title("🦅 Argus | Mission Control")

if st.button("🔄 Force Refresh"): st.rerun()

# 1. LIVE DATA FETCH
cash, btc_bal, sys_status = get_live_data()
live_price = get_live_ticker() 
history_df = scrape_equity_curve() 

if live_price == 0: live_price = 87150.00

# 2. STATUS BAR
if sys_status == "CONNECTED": 
    st.success(f"🟢 **SYSTEM ONLINE** | Market Feed Active")
else: 
    st.error(f"🔴 **SYSTEM ERROR** | {sys_status}")

st.markdown("---")

# 3. METRICS
equity = cash + (btc_bal * live_price)
crypto_val = btc_bal * live_price
total_return = ((equity - 100.00) / 100.00) * 100 

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net Liquid Equity", f"${equity:,.2f}", f"{total_return:.2f}%")
c2.metric("Dry Powder (USD)", f"${cash:,.2f}")
c3.metric("BTC Exposure", f"${crypto_val:,.2f}", f"{btc_bal:.6f} BTC")
c4.metric("Live Market Price", f"${live_price:,.2f}")

st.markdown("---")

# 4. VISUALIZATION
col_chart, col_brain = st.columns([2, 1])
with col_chart:
    st.subheader("Performance Curve")
    if history_df is not None and not history_df.empty:
        st.plotly_chart(create_equity_chart(history_df), use_container_width=True)
    else: 
        st.info("Initializing Data Stream... (Chart will appear after first trade)")
        st.plotly_chart(create_equity_chart(None), use_container_width=True)

with col_brain:
    st.subheader("Cortex State")
    st.plotly_chart(make_gauge(0.50), use_container_width=True) 
    st.caption(f"Strategy Signal: **WAITING**")

# 5. SCROLLABLE TERMINAL LOGS
st.markdown("---")
st.subheader("📜 System Logs (Live Stream)")

log_content = "Waiting for logs..."

# --- FIXED: Reading the VERBOSE EXECUTION_LOG ---
if EXECUTION_LOG.exists():
    try:
        if EXECUTION_LOG.stat().st_size > 0:
            with open(EXECUTION_LOG, "r", encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                log_content = "".join(lines[-50:]) 
            
            st.code(log_content, language='log')
            
        else:
            st.info("Execution log found, but is currently empty.")

    except Exception as e:
        st.error(f"Error reading log file: {e}")
else:
    st.info(log_content)

# Auto-Refresh
time.sleep(30)
st.rerun()