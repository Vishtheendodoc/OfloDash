import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import time

# --- Sidebar Controls ---
st.sidebar.title("Order Flow Controls")

# Fetch stock list
@st.cache_data
def get_stocks():
    try:
        response = requests.get("http://localhost:5000/api/stocks", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching stocks: {e}")
        return []

stocks = get_stocks()
if not stocks:
    st.stop()

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
    st.session_state['last_fetch_time'] = 0

# Store the selected interval in session state
if 'current_interval' not in st.session_state:
    st.session_state['current_interval'] = interval

# --- Fetch Data with improved error handling ---
def fetch_delta_data(security_id, interval):
    """Fetch delta data with proper error handling and validation"""
    url = f"http://localhost:5000/api/delta_data/{security_id}?interval={interval}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Validate data structure
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Validate required columns
        required_columns = ['timestamp', 'buy_volume', 'sell_volume']
        if not all(col in df.columns for col in required_columns):
            st.warning(f"Missing required columns in data: {required_columns}")
            return pd.DataFrame()
        
        # Filter out rows with all zero volumes (likely invalid data)
        df = df[~((df['buy_volume'] == 0) & (df['sell_volume'] == 0))]
        
        return df
        
    except requests.exceptions.RequestException as e:
        st.error(f"Network error fetching data: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing data: {e}")
        return pd.DataFrame()

# --- Main Data Processing ---
current_time = time.time()

# Only fetch if enough time has passed (prevent too frequent requests)
if current_time - st.session_state.get('last_fetch_time', 0) > 4:  # 4 seconds minimum between fetches
    df = fetch_delta_data(selected_id, interval)
    st.session_state['last_fetch_time'] = current_time
    
    if not df.empty:
        # Remove duplicates based on timestamp
        if not st.session_state['all_data'].empty:
            # Get existing timestamps
            existing_timestamps = set(st.session_state['all_data']['timestamp'].values)
            df = df[~df['timestamp'].isin(existing_timestamps)]
        
        # Only add if there's new data
        if not df.empty:
            st.session_state['all_data'] = pd.concat([st.session_state['all_data'], df], ignore_index=True)
            st.session_state['all_data'].sort_values('timestamp', inplace=True)
            st.session_state['all_data'].reset_index(drop=True, inplace=True)

# --- Data Aggregation Based on Selected Interval ---
def aggregate_data_by_interval(df, interval_minutes):
    """Aggregate the raw data based on the selected interval"""
    if df.empty:
        return df
    
    # Convert timestamp to datetime if it's not already
    df_copy = df.copy()
    df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
    
    # Create interval groups
    df_copy['interval_group'] = df_copy['timestamp'].dt.floor(f'{interval_minutes}min')
    
    # Aggregate by interval
    aggregated = df_copy.groupby('interval_group').agg({
        'buy_volume': 'sum',
        'sell_volume': 'sum'
    }).reset_index()
    
    # Rename the timestamp column back
    aggregated = aggregated.rename(columns={'interval_group': 'timestamp'})
    
    # Calculate delta and cumulative delta
    aggregated['delta'] = aggregated['buy_volume'] - aggregated['sell_volume']
    aggregated['cumulative_delta'] = aggregated['delta'].cumsum()
    
    return aggregated

# Get the display data based on selected interval
display_data = aggregate_data_by_interval(st.session_state['all_data'], interval)

# Update current interval in session state
st.session_state['current_interval'] = interval

# --- Display Data ---
if not display_data.empty:
    # --- Main Area ---
    st.title(f"Order Flow Dashboard: {selected_label}")
    
    last_update = display_data['timestamp'].iloc[-1]
    data_count = len(display_data)
    raw_data_count = len(st.session_state['all_data'])
    
    st.caption(f"Interval: {interval} min | Last update: {last_update} | Aggregated records: {data_count} | Raw records: {raw_data_count}")

    # Stock Info
    latest_buy = display_data['buy_volume'].iloc[-1]
    latest_sell = display_data['sell_volume'].iloc[-1]
    latest_delta = display_data['delta'].iloc[-1]
    cumulative_delta = display_data['cumulative_delta'].iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Latest Buy", f"{latest_buy:,.0f}")
    with col2:
        st.metric("Latest Sell", f"{latest_sell:,.0f}")
    with col3:
        st.metric("Latest Delta", f"{latest_delta:,.0f}")
    with col4:
        st.metric("Cumulative Delta", f"{cumulative_delta:,.0f}")

    # Charts
    st.subheader("Buy & Sell Volume")
    chart_data = display_data.set_index('timestamp')[['buy_volume', 'sell_volume']]
    st.line_chart(chart_data)

    st.subheader("Delta (Buy - Sell)")
    delta_data = display_data.set_index('timestamp')['delta']
    st.bar_chart(delta_data)

    st.subheader("Cumulative Delta")
    cumulative_data = display_data.set_index('timestamp')['cumulative_delta']
    st.line_chart(cumulative_data)

    # Data Table & Download
    st.subheader("Aggregated Data")
    
    # Show all aggregated data
    st.dataframe(display_data, use_container_width=True)
    
    # Download button for aggregated data
    csv = display_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Aggregated CSV",
        data=csv,
        file_name=f"orderflow_aggregated_{selected_id}_{interval}min.csv",
        mime="text/csv"
    )
    
    # Raw data section
    with st.expander("Raw Data (All Records)"):
        st.dataframe(st.session_state['all_data'], use_container_width=True)
        raw_csv = st.session_state['all_data'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Raw CSV",
            data=raw_csv,
            file_name=f"orderflow_raw_{selected_id}.csv",
            mime="text/csv",
            key="raw_download"
        )
    
    # Debug info in expander
    with st.expander("Debug Info"):
        st.write(f"Raw records: {len(st.session_state['all_data'])}")
        st.write(f"Aggregated records: {len(display_data)}")
        st.write(f"Selected interval: {interval} minutes")
        st.write(f"Last fetch time: {time.strftime('%H:%M:%S', time.localtime(st.session_state.get('last_fetch_time', 0)))}")
        st.write(f"Current time: {time.strftime('%H:%M:%S')}")
        st.write(f"Auto-refresh: {auto_refresh}")

else:
    st.info("No data available yet for this stock and interval. Waiting for data...")
    if auto_refresh:
        st.info("Auto-refresh is enabled. Data will appear automatically when available.")
    else:
        st.info("Enable auto-refresh or manually refresh the page to fetch data.")
