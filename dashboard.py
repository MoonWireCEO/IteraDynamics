# dashboard.py
# 🦅 ARGUS MISSION CONTROL - V3.3 (RESTORED TO V2.3 BASELINE)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time, requests, json, sys, os
from pathlib import Path
from dotenv import load_dotenv

# --- PATH SETUP (V2.3 Original) ---
current_file = Path(__file__).resolve()
project_root = current_file.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

# --- IMPORT LIVE BROKER (V2.3 Original) ---
try:
    from src.real_broker import RealBroker
except ImportError:
    st.error("❌ Could not import RealBroker. Check path.")
    st.stop()

st.set_page_config(page_title="Argus Commander", page_icon="🦅", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM STYLING (V2.3 Original CSS) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-family: 'Monospace'; }
    [data-testid="stMetricValue"] { font-size: 26px; font-family: 'Monospace'; color: #00ff00; }
    div[data-testid="stMetricLabel"] > label { color: #ffffff !important; font-size: 14px; font-weight: bold; }
    .log-box {
        font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.5;
        color: #00ff00 !important; background-color: #000000; padding: 15px;
        border: 1px solid #333; border-radius: 5px; height: 350px; overflow-y: scroll;
        white-space: pre-wrap;
    }
    .log-box * { color: #00ff00 !important; opacity: 1 !important; }
</style>
""", unsafe_allow_html=True)

# --- LIVE DATA FUNCTIONS ---

@st.cache_resource
def get_broker(): return RealBroker()

def get_live_price():
    try:
        url = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
        resp = requests.get(url, timeout=3)
        return float(resp.json()['data']['amount'])
    except: return 0.0

def load_market_data():
    csv_path = project_root / "flight_recorder.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if 'Close' not in df.columns:
                 df = pd.read_csv(csv_path, names=["Timestamp","Open","High","Low","Close","Volume"], header=0)
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df.sort_values('Timestamp', inplace=True)
            return df
        except: pass
    return pd.DataFrame()

def read_logs():
    log_path = project_root / "argus.log"
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                return "".join(f.readlines()[-300:])
        except: return "Error reading log."
    return "Waiting for logs..."

# --- 🎯 AUTOMATION UPDATES ---

def get_cortex_state():
    """Reads Risk Multiplier from logs to drive the gauge."""
    log_path = project_root / "argus.log"
    state = {"conviction_score": 0, "regime": "Searching...", "risk_mult": 0.0}
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                for line in reversed(lines[-100:]):
                    if "[REGIME]" in line and "Risk Multiplier:" in line:
                        regime = line.split("[REGIME]")[1].split("|")[0].strip()
                        risk_val = float(line.split("Risk Multiplier:")[1].strip())
                        state["regime"], state["risk_mult"] = regime, risk_val
                        state["conviction_score"] = int(risk_val * 100)
                        break
        except: pass
    return state

def get_auto_entry():
    """Pulls from trade_state.json for the dashboard metrics."""
    path = project_root / "trade_state.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return float(json.load(f).get("entry_price", 90163.00))
        except: pass
    return 90163.00

# --- MAIN DASHBOARD (V2.3 Structure) ---
def main():
    st.title("🦅 ARGUS // LIVE COMMANDER")
    try:
        broker = get_broker()
        current_price = get_live_price()
        df = load_market_data()
        avg_entry = get_auto_entry()
        equity = broker.cash + (broker.positions * current_price)

        if broker.positions > 0:
            unrealized_pnl_usd = (current_price - avg_entry) * broker.positions
            pnl_pct = (unrealized_pnl_usd / (avg_entry * broker.positions)) * 100
            breakeven = avg_entry * 1.002
        else:
            unrealized_pnl_usd, pnl_pct, breakeven = 0.0, 0.0, 0.0
    except Exception as e:
        st.error(f"System Error: {e}"); return

    # --- ROW 1: STATUS ---
    st.markdown("### 🏦 Liquid Status")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Net Liquid Equity", f"${equity:,.2f}", delta=f"{unrealized_pnl_usd:+.2f}")
    k2.metric("Dry Powder (USD)", f"${broker.cash:,.2f}")
    k3.metric("BTC Exposure", f"${(broker.positions * current_price):,.2f}", f"{broker.positions:.6f} BTC")
    k4.metric("Market Price", f"${current_price:,.2f}")

    # --- ROW 2: ANALYSIS ---
    st.markdown("### 📊 Active Position Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Auto-Entry Price", f"${avg_entry:,.2f}")
    m2.metric("Unrealized P&L ($)", f"${unrealized_pnl_usd:+.2f}")
    color_mode = "normal" if pnl_pct >= 0 else "inverse"
    m3.metric("Unrealized P&L (%)", f"{pnl_pct:+.2f}%", delta=pnl_pct, delta_color=color_mode)
    m4.metric("Breakeven Price", f"${breakeven:,.2f}")

    st.markdown("---")

    # --- ROW 3: CHARTS & GAUGE ---
    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader("Performance Curve")
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Close'], mode='lines', line=dict(color='#00ff00', width=2)))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Cortex State")
        cortex = get_cortex_state()
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = cortex['conviction_score'],
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"},
                     'steps': [{'range': [0, 40], 'color': "red"}, {'range': [60, 100], 'color': "green"}]}
        ))
        fig_gauge.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(f"<p style='text-align: center; color: gray;'>Regime: {cortex['regime']}</p>", unsafe_allow_html=True)

    # --- ROW 4: LOGS ---
    st.subheader("📜 System Logs")
    st.markdown(f"<div class='log-box'>{read_logs()}</div>", unsafe_allow_html=True)

    time.sleep(10); st.rerun()

if __name__ == "__main__": main()