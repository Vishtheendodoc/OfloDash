import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import time
import json
from datetime import datetime, timedelta

# --- Configuration ---
st.set_page_config(
    page_title="Order Flow Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Sidebar Controls ---
st.sidebar.title("Order Flow Controls")

# Fetch stock list
@st.cache_data
def get_stocks():
    try:
        response = requests.get("https://oflo.onrender.com/api/stocks", timeout=10)
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

# Data management buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    reset_data = st.button("Reset Data")
with col2:
    clear_old_data = st.button("Clear Old Data")

# Data retention settings
st.sidebar.subheader("Data Management")
max_days = st.sidebar.slider("Keep data for (days)", 1, 30, 7)
max_records = st.sidebar.slider("Max records per stock", 1000, 50000, 10000)

# --- Auto-refresh ---
if auto_refresh:
    st_autorefresh(interval=5000, key="datarefresh")

# --- Initialize Session State ---
if 'persistent_data' not in st.session_state:
    st.session_state['persistent_data'] = {}

if 'last_fetch_times' not in st.session_state:
    st.session_state['last_fetch_times'] = {}

if 'fetch_errors' not in st.session_state:
    st.session_state['fetch_errors'] = {}

# --- Data Management Functions ---
def clean_old_data(stock_id, max_days, max_records):
    """Clean old data based on retention settings"""
    if stock_id not in st.session_state['persistent_data']:
        return
    
    df = st.session_state['persistent_data'][stock_id]
    if df.empty:
        return
    
    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Remove data older than max_days
    cutoff_date = datetime.now() - timedelta(days=max_days)
    df = df[df['timestamp'] >= cutoff_date]
    
    # Keep only the most recent max_records
    if len(df) > max_records:
        df = df.tail(max_records)
    
    st.session_state['persistent_data'][stock_id] = df

def save_data_to_persistent_storage(stock_id, new_data):
    """Save data to persistent storage with deduplication"""
    if stock_id not in st.session_state['persistent_data']:
        st.session_state['persistent_data'][stock_id] = pd.DataFrame()
    
    existing_data = st.session_state['persistent_data'][stock_id]
    
    if not new_data.empty:
        # Convert timestamps to ensure consistency
        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        
        if not existing_data.empty:
            existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])
            # Remove duplicates based on timestamp
            existing_timestamps = set(existing_data['timestamp'].values)
            new_data = new_data[~new_data['timestamp'].isin(existing_timestamps)]
        
        if not new_data.empty:
            # Combine and sort
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            combined_data.sort_values('timestamp', inplace=True)
            combined_data.reset_index(drop=True, inplace=True)
            
            # Store back
            st.session_state['persistent_data'][stock_id] = combined_data
            
            # Clean old data
            clean_old_data(stock_id, max_days, max_records)
            
            return True
    return False

def get_persistent_data(stock_id):
    """Get data from persistent storage"""
    if stock_id in st.session_state['persistent_data']:
        return st.session_state['persistent_data'][stock_id].copy()
    return pd.DataFrame()

# --- Data Fetching Functions ---
def fetch_delta_data(security_id, interval):
    """Fetch delta data with proper error handling and validation"""
    url = f"https://oflo.onrender.com/api/delta_data/{security_id}?interval={interval}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Clear previous errors
        if security_id in st.session_state['fetch_errors']:
            del st.session_state['fetch_errors'][security_id]
        
        # Validate data structure
        if not isinstance(data, list):
            return pd.DataFrame()
        
        if len(data) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Validate required columns
        required_columns = ['timestamp', 'buy_volume', 'sell_volume']
        if not all(col in df.columns for col in required_columns):
            st.warning(f"Missing required columns in data: {required_columns}")
            return pd.DataFrame()
        
        # Convert to proper data types
        df['buy_volume'] = pd.to_numeric(df['buy_volume'], errors='coerce').fillna(0)
        df['sell_volume'] = pd.to_numeric(df['sell_volume'], errors='coerce').fillna(0)
        
        # Filter out rows with invalid data
        df = df[~((df['buy_volume'] == 0) & (df['sell_volume'] == 0))]
        
        return df
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        st.session_state['fetch_errors'][security_id] = error_msg
        return pd.DataFrame()
    except Exception as e:
        error_msg = f"Error processing data: {str(e)}"
        st.session_state['fetch_errors'][security_id] = error_msg
        return pd.DataFrame()

# --- Data Aggregation Functions ---
def aggregate_data_by_interval(df, interval_minutes):
    """Aggregate the raw data based on the selected interval"""
    if df.empty:
        return df
    
    # Ensure timestamp is datetime
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

# --- Handle Data Reset/Clear ---
if reset_data:
    if selected_id in st.session_state['persistent_data']:
        del st.session_state['persistent_data'][selected_id]
    st.success("Data reset successfully!")
    st.rerun()

if clear_old_data:
    clean_old_data(selected_id, max_days, max_records)
    st.success("Old data cleared successfully!")
    st.rerun()

# --- Main Data Processing ---
current_time = time.time()
last_fetch_time = st.session_state['last_fetch_times'].get(selected_id, 0)

# Fetch new data (rate limited)
if current_time - last_fetch_time > 4:  # 4 seconds minimum between fetches
    new_data = fetch_delta_data(selected_id, 1)  # Always fetch at 1-minute intervals
    st.session_state['last_fetch_times'][selected_id] = current_time
    
    if not new_data.empty:
        data_updated = save_data_to_persistent_storage(selected_id, new_data)
        if data_updated:
            st.success(f"Updated with {len(new_data)} new records", icon="✅")

# Get all stored data for this stock
all_data = get_persistent_data(selected_id)

# Get the display data based on selected interval
display_data = aggregate_data_by_interval(all_data, interval)

# --- Display Data ---
if not display_data.empty:
    # --- Main Area ---
    st.title(f"📊 Order Flow Dashboard: {selected_label}")
    
    last_update = display_data['timestamp'].iloc[-1]
    data_count = len(display_data)
    raw_data_count = len(all_data)
    
    # Status indicators
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(f"Interval: {interval} min | Last update: {last_update}")
    with col2:
        st.caption(f"📈 Aggregated: {data_count}")
    with col3:
        st.caption(f"📋 Raw: {raw_data_count}")
    
    # Show any fetch errors
    if selected_id in st.session_state['fetch_errors']:
        st.error(f"⚠️ {st.session_state['fetch_errors'][selected_id]}")

    # Metrics
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
        delta_color = "normal" if latest_delta >= 0 else "inverse"
        st.metric("Latest Delta", f"{latest_delta:,.0f}")
    with col4:
        cumulative_color = "normal" if cumulative_delta >= 0 else "inverse"
        st.metric("Cumulative Delta", f"{cumulative_delta:,.0f}")

    # Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📊 Buy & Sell Volume")
        chart_data = display_data.set_index('timestamp')[['buy_volume', 'sell_volume']]
        st.line_chart(chart_data, use_container_width=True)
    
    with chart_col2:
        st.subheader("📈 Cumulative Delta")
        cumulative_data = display_data.set_index('timestamp')['cumulative_delta']
        st.line_chart(cumulative_data, use_container_width=True)

    st.subheader("📊 Delta (Buy - Sell)")
    delta_data = display_data.set_index('timestamp')['delta']
    st.bar_chart(delta_data, use_container_width=True)

    # Data Tables
    tab1, tab2, tab3 = st.tabs(["📋 Aggregated Data", "📊 Raw Data", "⚙️ Debug Info"])
    
    with tab1:
        st.dataframe(display_data, use_container_width=True, height=400)
        
        # Download aggregated data
        csv = display_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Aggregated CSV",
            data=csv,
            file_name=f"orderflow_aggregated_{selected_id}_{interval}min_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.dataframe(all_data, use_container_width=True, height=400)
        
        # Download raw data
        if not all_data.empty:
            raw_csv = all_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Raw CSV",
                data=raw_csv,
                file_name=f"orderflow_raw_{selected_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="raw_download"
            )
    
    with tab3:
        st.write("**Data Statistics:**")
        st.write(f"- Raw records: {len(all_data)}")
        st.write(f"- Aggregated records: {len(display_data)}")
        st.write(f"- Selected interval: {interval} minutes")
        st.write(f"- Data retention: {max_days} days, {max_records} max records")
        
        st.write("**System Status:**")
        st.write(f"- Last fetch: {time.strftime('%H:%M:%S', time.localtime(last_fetch_time))}")
        st.write(f"- Current time: {time.strftime('%H:%M:%S')}")
        st.write(f"- Auto-refresh: {auto_refresh}")
        st.write(f"- Memory usage: {len(st.session_state['persistent_data'])} stocks tracked")
        
        # Show data age
        if not all_data.empty:
            oldest_data = all_data['timestamp'].min()
            newest_data = all_data['timestamp'].max()
            st.write(f"- Data range: {oldest_data} to {newest_data}")

else:
    st.info("📊 No data available yet for this stock and interval.")
    st.info("🔄 Data is being collected and stored locally in Streamlit...")
    
    if selected_id in st.session_state['fetch_errors']:
        st.error(f"⚠️ Backend Error: {st.session_state['fetch_errors'][selected_id]}")
    
    if auto_refresh:
        st.info("🔄 Auto-refresh is enabled. Data will appear automatically when available.")
    else:
        st.info("🔄 Enable auto-refresh to continuously collect data.")

# --- Sidebar Status ---
st.sidebar.subheader("📊 Data Status")
total_stocks = len(st.session_state['persistent_data'])
total_records = sum(len(df) for df in st.session_state['persistent_data'].values())
st.sidebar.metric("Tracked Stocks", total_stocks)
st.sidebar.metric("Total Records", total_records)

# Show storage for each stock
if st.sidebar.checkbox("Show Stock Details"):
    for stock_id, df in st.session_state['persistent_data'].items():
        if not df.empty:
            oldest = df['timestamp'].min()
            newest = df['timestamp'].max()
            st.sidebar.write(f"**{stock_id}**: {len(df)} records")
            st.sidebar.caption(f"From {oldest} to {newest}")
