import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- Session State Initialization ---
if 'persistent_data' not in st.session_state:
    st.session_state['persistent_data'] = {}
if 'last_fetch_times' not in st.session_state:
    st.session_state['last_fetch_times'] = {}
if 'fetch_errors' not in st.session_state:
    st.session_state['fetch_errors'] = {}

# --- Data Fetching Function ---
def fetch_delta_data(security_id, interval):
    """Fetch delta data with proper error handling and validation"""
    url = f"https://oflo.onrender.com/api/delta_data/{security_id}?interval={interval}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Validate data structure
        if not isinstance(data, list):
            return pd.DataFrame()
        if len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

# --- Data Persistence Function ---
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
            return True
    return False

# --- Data Retrieval Function ---
def get_persistent_data(stock_id):
    """Get data from persistent storage"""
    if stock_id in st.session_state['persistent_data']:
        return st.session_state['persistent_data'][stock_id].copy()
    return pd.DataFrame()

# --- Data Aggregation Function ---
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
    agg_dict = {
        'buy_volume': 'sum',
        'sell_volume': 'sum'
    }
    
    # Add OHLC aggregation if price data exists
    if 'open' in df_copy.columns:
        agg_dict.update({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
    
    # Add tick-rule aggregation if it exists
    if 'buy_initiated' in df_copy.columns:
        agg_dict.update({
            'buy_initiated': 'sum',
            'sell_initiated': 'sum'
        })
    
    aggregated = df_copy.groupby('interval_group').agg(agg_dict).reset_index()
    
    # Rename the timestamp column back
    aggregated = aggregated.rename(columns={'interval_group': 'timestamp'})
    
    # Calculate delta and cumulative delta
    aggregated['delta'] = aggregated['buy_volume'] - aggregated['sell_volume']
    aggregated['cumulative_delta'] = aggregated['delta'].cumsum()
    
    # Calculate tick delta and inference if tick data exists
    if 'buy_initiated' in aggregated.columns:
        aggregated['tick_delta'] = aggregated['buy_initiated'] - aggregated['sell_initiated']
        # Simple inference logic
        aggregated['inference'] = aggregated['tick_delta'].apply(
            lambda x: 'Buy Dominant' if x > 0 else ('Sell Dominant' if x < 0 else 'Neutral')
        )
    
    return aggregated

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
    st.session_state['current_stock'] = selected_id
    st.session_state['last_fetch_times'] = {}

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

# Always get the latest raw data for the selected stock
all_data = get_persistent_data(selected_id)

# Always aggregate for the current interval
display_data = aggregate_data_by_interval(all_data, interval)

# --- Main Display Logic ---
if not display_data.empty:
    # --- Main Area ---
    st.title(f"Order Flow Dashboard: {selected_label}")
    st.caption(f"Interval: {interval} min | Last update: {display_data['timestamp'].iloc[-1]}")

    # Stock Info
    st.markdown(f"**Latest Buy:** {display_data['buy_volume'].iloc[-1]} | "
                f"**Latest Sell:** {display_data['sell_volume'].iloc[-1]} | "
                f"**Latest Delta:** {display_data['delta'].iloc[-1]} | "
                f"**Cumulative Delta:** {display_data['cumulative_delta'].iloc[-1]}")
    
    # Tick-rule summary
    if 'buy_initiated' in display_data.columns:
        st.markdown(f"**Latest Buy-Initiated:** {display_data['buy_initiated'].iloc[-1]} | "
                    f"**Latest Sell-Initiated:** {display_data['sell_initiated'].iloc[-1]} | "
                    f"**Latest Tick Delta:** {display_data['tick_delta'].iloc[-1]} | "
                    f"**Inference:** {display_data['inference'].iloc[-1]}")

    # --- Enhanced Candlestick Chart with Volume Labels ---
    if all(col in display_data.columns for col in ['open', 'high', 'low', 'close']):
        st.subheader("Candlestick Chart with Order Flow")
        ohlc_df = display_data.dropna(subset=['open', 'high', 'low', 'close'])
        if not ohlc_df.empty:
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=ohlc_df['timestamp'],
                open=ohlc_df['open'],
                high=ohlc_df['high'],
                low=ohlc_df['low'],
                close=ohlc_df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                increasing_fillcolor='#26a69a',
                decreasing_fillcolor='#ef5350'
            ))
            
            # Add volume labels and inference markers
            for i, row in ohlc_df.iterrows():
                # Calculate position for volume labels
                candle_range = row['high'] - row['low']
                candle_mid = (row['high'] + row['low']) / 2
                
                # Buy initiated volume (green, above candle)
                if 'buy_initiated' in ohlc_df.columns and pd.notna(row['buy_initiated']):
                    fig.add_trace(go.Scatter(
                        x=[row['timestamp']],
                        y=[row['high'] + candle_range * 0.15],
                        mode='text',
                        text=[f"B: {int(row['buy_initiated'])}"],
                        textfont=dict(color='#26a69a', size=10, family='Arial Black'),
                        textposition='middle center',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Sell initiated volume (red, below candle)
                if 'sell_initiated' in ohlc_df.columns and pd.notna(row['sell_initiated']):
                    fig.add_trace(go.Scatter(
                        x=[row['timestamp']],
                        y=[row['low'] - candle_range * 0.15],
                        mode='text',
                        text=[f"S: {int(row['sell_initiated'])}"],
                        textfont=dict(color='#ef5350', size=10, family='Arial Black'),
                        textposition='middle center',
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                # Net delta inside the candle (if significant)
                if 'tick_delta' in ohlc_df.columns and pd.notna(row['tick_delta']):
                    delta_val = int(row['tick_delta'])
                    if abs(delta_val) > 0:  # Only show if there's a net difference
                        delta_color = 'black'
                        fig.add_trace(go.Scatter(
                            x=[row['timestamp']],
                            y=[candle_mid],
                            mode='text',
                            text=[f"Δ{delta_val:+}"],
                            textfont=dict(color=delta_color, size=9, family='Arial Black'),
                            textposition='middle center',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                
                # Inference arrows (optional, can be commented out if too cluttered)
                if 'inference' in ohlc_df.columns and pd.notna(row['inference']):
                    if row['inference'] == 'Buy Dominant':
                        fig.add_trace(go.Scatter(
                            x=[row['timestamp']],
                            y=[row['high'] + candle_range * 0.35],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                color='#26a69a',
                                size=12,
                                line=dict(width=1, color='white')
                            ),
                            name='Buy Dominant' if i == 0 else None,
                            showlegend=i == 0,
                            hovertemplate='<b>Buy Dominant</b><br>Time: %{x}<br>Price: %{y}<extra></extra>'
                        ))
                    elif row['inference'] == 'Sell Dominant':
                        fig.add_trace(go.Scatter(
                            x=[row['timestamp']],
                            y=[row['low'] - candle_range * 0.35],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                color='#ef5350',
                                size=12,
                                line=dict(width=1, color='white')
                            ),
                            name='Sell Dominant' if i == 0 else None,
                            showlegend=i == 0,
                            hovertemplate='<b>Sell Dominant</b><br>Time: %{x}<br>Price: %{y}<extra></extra>'
                        ))
            
            # Update layout
            fig.update_layout(
                title=f'Order Flow Analysis - {selected_label}',
                xaxis_title='Time',
                yaxis_title='Price',
                height=600,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(t=80, b=50, l=50, r=50),
                plot_bgcolor='white',
                xaxis=dict(
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    zeroline=False
                ),
                yaxis=dict(
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    zeroline=False
                )
            )
            
            # Remove range slider for cleaner look
            fig.update_layout(xaxis_rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            #st.caption("""
            #📊 **Chart Legend:**
            #- **Green triangles ▲**: Buy dominant periods
            #- **Red triangles ▼**: Sell dominant periods  
            #- **B: [number]**: Buy initiated volume (above candles)
            #- **S: [number]**: Sell initiated volume (below candles)
            #- **Δ[+/-number]**: Net tick delta inside candles
            #""")

    # --- Alternative: Volume Bar Chart Overlay ---
    if st.checkbox("Show Volume Bar Overlay", value=False):
        if all(col in display_data.columns for col in ['buy_initiated', 'sell_initiated']):
            st.subheader("Volume Analysis")
            
            # Create subplot with secondary y-axis
            from plotly.subplots import make_subplots
            
            fig_vol = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price Action', 'Order Flow Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Add candlestick to upper subplot
            fig_vol.add_trace(
                go.Candlestick(
                    x=ohlc_df['timestamp'],
                    open=ohlc_df['open'],
                    high=ohlc_df['high'],
                    low=ohlc_df['low'],
                    close=ohlc_df['close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
            
            # Add volume bars to lower subplot
            fig_vol.add_trace(
                go.Bar(
                    x=ohlc_df['timestamp'],
                    y=ohlc_df['buy_initiated'],
                    name='Buy Initiated',
                    marker_color='#26a69a',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig_vol.add_trace(
                go.Bar(
                    x=ohlc_df['timestamp'],
                    y=-ohlc_df['sell_initiated'],  # Negative for visual separation
                    name='Sell Initiated',
                    marker_color='#ef5350',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Update layout
            fig_vol.update_layout(
                height=700,
                showlegend=True,
                title_text="Price Action with Order Flow Volume"
            )
            
            # Update y-axis labels
            fig_vol.update_yaxes(title_text="Price", row=1, col=1)
            fig_vol.update_yaxes(title_text="Volume", row=2, col=1)
            fig_vol.update_xaxes(title_text="Time", row=2, col=1)
            
            st.plotly_chart(fig_vol, use_container_width=True)
    # Charts - Now using display_data instead of st.session_state['all_data']
    #st.subheader("Buy & Sell Volume")
    #st.line_chart(display_data.set_index('timestamp')[['buy_volume', 'sell_volume']])

    #st.subheader("Delta (Buy - Sell)")
    #st.bar_chart(display_data.set_index('timestamp')['delta'])

    #st.subheader("Cumulative Delta")
    #st.line_chart(display_data.set_index('timestamp')['cumulative_delta'])

    # Data Table & Download - Now using display_data
    st.subheader("Raw Data")
    # Show all columns including tick-rule order flow
    display_cols = [
        'timestamp', 'buy_volume', 'sell_volume', 'delta', 'cumulative_delta',
        'buy_initiated', 'sell_initiated', 'tick_delta', 'inference'
    ]
    display_cols = [col for col in display_cols if col in display_data.columns]
    st.dataframe(display_data[display_cols])
    csv = display_data[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "orderflow_data.csv", "text/csv")
else:
    st.info("No data yet for this stock and interval.")
