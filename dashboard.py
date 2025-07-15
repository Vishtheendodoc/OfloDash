import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# --- Sidebar Controls ---
st.sidebar.title("Order Flow Controls")

# Fetch stock list
@st.cache_data
def get_stocks():
    return requests.get("https://oflo.onrender.com/api/stocks").json()

stocks = get_stocks()
stock_options = {f"{s['symbol']} ({s['security_id']})": s['security_id'] for s in stocks}
selected_label = st.sidebar.selectbox("Select Stock", list(stock_options.keys()))
selected_id = stock_options[selected_label]
interval = st.sidebar.selectbox("Interval (minutes)", [1, 3, 5, 15, 20, 30])
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
reset_data = st.sidebar.button("Reset Data")

# --- Auto-refresh ---
if auto_refresh:
    st_autorefresh(interval=5000, key="datarefresh")

# --- Session State for Data ---
if 'current_stock' not in st.session_state or reset_data or st.session_state.get('current_stock') != selected_id:
    st.session_state['all_data'] = pd.DataFrame()
    st.session_state['current_stock'] = selected_id

# --- Fetch Data ---
url = f"http://localhost:5000/api/delta_data/{selected_id}?interval={interval}"
try:
    bars = requests.get(url).json()
    df = pd.DataFrame(bars)
    if not df.empty:
        # Remove duplicates based on timestamp
        if not st.session_state['all_data'].empty:
            df = df[~df['timestamp'].isin(st.session_state['all_data']['timestamp'])]
        st.session_state['all_data'] = pd.concat([st.session_state['all_data'], df], ignore_index=True)
        st.session_state['all_data'].sort_values('timestamp', inplace=True)
        st.session_state['all_data'].reset_index(drop=True, inplace=True)

        # Calculate delta and cumulative delta
        st.session_state['all_data']['delta'] = st.session_state['all_data']['buy_volume'] - st.session_state['all_data']['sell_volume']
        st.session_state['all_data']['cumulative_delta'] = st.session_state['all_data']['delta'].cumsum()

        # --- Main Area ---
        st.title(f"Order Flow Dashboard: {selected_label}")
        st.caption(f"Interval: {interval} min | Last update: {st.session_state['all_data']['timestamp'].iloc[-1]}")

        # Stock Info
        st.markdown(f"**Latest Buy:** {st.session_state['all_data']['buy_volume'].iloc[-1]} | "
                    f"**Latest Sell:** {st.session_state['all_data']['sell_volume'].iloc[-1]} | "
                    f"**Latest Delta:** {st.session_state['all_data']['delta'].iloc[-1]} | "
                    f"**Cumulative Delta:** {st.session_state['all_data']['cumulative_delta'].iloc[-1]}")

        # Charts
        st.subheader("Buy & Sell Volume")
        st.line_chart(st.session_state['all_data'].set_index('timestamp')[['buy_volume', 'sell_volume']])

        st.subheader("Delta (Buy - Sell)")
        st.bar_chart(st.session_state['all_data'].set_index('timestamp')['delta'])

        st.subheader("Cumulative Delta")
        st.line_chart(st.session_state['all_data'].set_index('timestamp')['cumulative_delta'])

        # Data Table & Download
        st.subheader("Raw Data")
        st.dataframe(st.session_state['all_data'])
        csv = st.session_state['all_data'].to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "orderflow_data.csv", "text/csv")
    else:
        st.info("No data yet for this stock and interval.")
except Exception as e:
    st.error(f"Error fetching delta data: {e}") 
