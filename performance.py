import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import weakref
import _thread
import math

# Google Sheets credentials setup
credentials_info = {
    "type": st.secrets["google_credentials"]["type"],
    "project_id": st.secrets["google_credentials"]["project_id"],
    "private_key_id": st.secrets["google_credentials"]["private_key_id"],
    "private_key": st.secrets["google_credentials"]["private_key"],
    "client_email": st.secrets["google_credentials"]["client_email"],
    "client_id": st.secrets["google_credentials"]["client_id"],
    "auth_uri": st.secrets["google_credentials"]["auth_uri"],
    "token_uri": st.secrets["google_credentials"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["google_credentials"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["google_credentials"]["client_x509_cert_url"]
}

scope = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Authenticate and access Google Sheets
credentials = Credentials.from_service_account_info(credentials_info, scopes=scope)
client = gspread.authorize(credentials)

def no_op_hash(obj):
    return str(obj)

def weak_method_hash(obj):
    return str(obj)

@st.cache_data(hash_funcs={_thread.RLock: no_op_hash, weakref.WeakMethod: weak_method_hash})
def read_gsheet_to_df(sheet_name, worksheet_name):
    try:
        spreadsheet = client.open(sheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Spreadsheet '{sheet_name}' not found.")
        return None

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Worksheet '{worksheet_name}' not found in spreadsheet '{sheet_name}'.")
        return None

    data = worksheet.get_all_records()
    return pd.DataFrame(data)

def write_to_gsheet(sheet_name, worksheet_name, new_row):
    """Append a new row of data to the Google Sheet."""
    try:
        spreadsheet = client.open(sheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"Spreadsheet '{sheet_name}' not found.")
        return None

    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        st.error(f"Worksheet '{worksheet_name}' not found in spreadsheet '{sheet_name}'.")
        return None

    worksheet.append_row(new_row)
    st.success("Data added successfully to the sheet!")

def fetch_data(sheet_name, worksheet_name):
    try:
        return read_gsheet_to_df(sheet_name, worksheet_name)
    except Exception as e:
        st.error(f"Failed to load {worksheet_name} into DataFrame: {e}")
        return None

def generate_supplier_id():
    sheet_name = "chip"
    worksheet_name = "agent_performance"

    # Clear both data and resource caches
    st.cache_data.clear()
    st.cache_resource.clear()

    survey_4 = read_gsheet_to_df(sheet_name, worksheet_name)

    last_supplier_id = survey_4.iloc[-1]["Supplier Id"] if survey_4 is not None and not survey_4.empty else "Supplier0000"

    match = re.search(r"Supplier(\d+)", last_supplier_id)
    if match:
        supplier_count = int(match.group(1)) + 1
    else:
        supplier_count = 1

    return f"Supplier{str(supplier_count).zfill(4)}"

def clean_location_name(location, filtered_survey):
    if not isinstance(location, str):
        location = str(location)
    cleaned_name = re.sub(r'benchmark location \d+', '', location)
    cleaned_name = re.sub(r'Distribution center \d+', '', cleaned_name)
    cleaned_name = cleaned_name.strip()
    count = filtered_survey[filtered_survey['Location'] == location].shape[0]
    return f"{cleaned_name} ({count})"

def concatenate_dfs(*dfs):
    return pd.concat(dfs, ignore_index=True)

sheets_and_worksheets = [
    ('chip', 'sunday'),
    ('chip', 'Localshops'),
    ('chip', 'Distribution'),
    ('chip', 'Farm_'),
    ('chip', 'agent_performance')    
]

data_frames = {}
with ThreadPoolExecutor() as executor:
    future_to_sheet = {executor.submit(fetch_data, sheet, worksheet): worksheet for sheet, worksheet in sheets_and_worksheets}
    for future in as_completed(future_to_sheet):
        worksheet_name = future_to_sheet[future]
        try:
            data_frames[worksheet_name] = future.result()
        except Exception as e:
            st.error(f"Failed to load data from {worksheet_name}: {e}")

try:
    survey_0 = data_frames.get('sunday')
    survey_1 = data_frames.get('Localshops')
    survey_2 = data_frames.get('Distribution')
    survey_3 = data_frames.get('Farm_')
    survey_4 = data_frames.get('agent_performance')
    if survey_2 is not None:
        survey_2 = survey_2.rename(columns={'Buying Price': 'Unit Price', 'Location ': 'Location', 'Product List': 'Products List'})
    if survey_3 is not None:
        survey_3 = survey_3.rename(columns={'Buying Price per Kg ': 'Unit Price', 'Product Origin ': 'Location', 'Product List': 'Products List', 'Farm Source Type': 'Farm_Source_Type'})

    survey_0.iloc[:, 0] = pd.to_datetime(survey_0.iloc[:, 0], format="%m/%d/%Y %H:%M:%S").dt.date
    survey_1.iloc[:, 2] = pd.to_datetime(survey_1.iloc[:, 2], format="%Y-%m-%d %H:%M:%S").dt.date
    survey_2.iloc[:, 0] = pd.to_datetime(survey_2.iloc[:, 0], format="%m/%d/%Y %H:%M:%S").dt.date
    survey_3.iloc[:, 0] = pd.to_datetime(survey_3.iloc[:, 0], format="%m/%d/%Y %H:%M:%S").dt.date

    survey = concatenate_dfs(survey_0, survey_1, survey_2, survey_3)
except Exception as e:
    st.error(f"Data processing failed: {e}")
    st.stop()
try:
    default_start = pd.to_datetime('today') - pd.to_timedelta(7, unit='d')
    default_end = pd.to_datetime('today')
    start_date = st.sidebar.date_input("From", value=default_start, key="unique_start_date")
    end_date = st.sidebar.date_input("To", value=default_end, key="unique_end_date")
    selected_date_range = (start_date, end_date)
except Exception as e:
    st.error(f'Error selecting date range: {e}')
    st.stop()    
try:
    filtered_survey = survey[(survey['Timestamp'] >= start_date) & (survey['Timestamp'] <= end_date)]
    available_products = filtered_survey['Products List'].unique()
    available_locations = filtered_survey['Location'].unique()
    selected_product =  st.sidebar.selectbox("Select Product", available_products, key='unique_key_2')
    end_date_data = survey[(survey['Products List'] == selected_product) & (survey['Timestamp'] == end_date)]
    combined = concatenate_dfs(survey)
   
except Exception as e:
    st.error(f'Failed to select product or group: {e}')
    st.stop()




# try:
#     cleaned_location_groups_with_counts = {group: [clean_location_name(loc, filtered_survey) for loc in locations] for group, locations in location_groups.items()}
#     reverse_location_mapping = {clean_location_name(loc, filtered_survey): loc for loc in survey['Location'].unique()}
#     all_sorted_locations = []
#     selected_groups_default = [list(location_groups.keys())[4], list(location_groups.keys())[2], list(location_groups.keys())[1]]
#     selected_groups = st.sidebar.multiselect("Select Location Groups for Comparison", options=list(location_groups.keys()), default=selected_groups_default)
# except Exception as e:
#     st.error(f'Failed to select location groups: {e}')
#     st.stop()

def calculate_benchmark_prices(filtered_survey, selected_date_range, selected_product):
    # Filter data based on the date range and selected product
    start_date, end_date = selected_date_range
    filtered_data = filtered_survey[
        (filtered_survey['Timestamp'] >= start_date) &
        (filtered_survey['Timestamp'] <= end_date) &
        (filtered_survey['Products List'] == selected_product)
    ]
    
    # Initialize dictionary to store minimum and average prices for each location group
    location_prices = {
        "Local Shops": [],
        "Supermarkets": [],
        "Sunday Markets": [],
        "Distribution Centers": [],
    }
    
    # Loop through the filtered data and classify prices
    for location in filtered_data["Location"].unique():
        if re.search(r'suk', location, re.IGNORECASE):
            location_prices["Local Shops"].append(filtered_data[filtered_data["Location"] == location]["Unit Price"].values)
        elif re.search(r'supermarket', location, re.IGNORECASE):
            location_prices["Supermarkets"].append(filtered_data[filtered_data["Location"] == location]["Unit Price"].values)
        elif re.search(r'sunday', location, re.IGNORECASE):
            location_prices["Sunday Markets"].append(filtered_data[filtered_data["Location"] == location]["Unit Price"].values)
        elif re.search(r'Distribution center', location, re.IGNORECASE):
            location_prices["Distribution Centers"].append(filtered_data[filtered_data["Location"] == location]["Unit Price"].values)

    # Calculate min and average prices
    benchmark_prices = {}
    for group, prices in location_prices.items():
        all_prices = [price for sublist in prices for price in sublist]
        if all_prices:  # If there are any prices
            benchmark_prices[group] = {
                "min_price": min(all_prices),
                "avg_price": sum(all_prices) / len(all_prices)
            }
    
    return benchmark_prices


def compare_farm_prices_with_benchmarks(farm_data, benchmark_prices, selected_date_range):
    comparison_results = []
    start_date, end_date = selected_date_range
    filtered_farm_data = farm_data[
        (farm_data['Timestamp'] >= start_date) &
        (farm_data['Timestamp'] <= end_date) &
        (farm_data['Products List'] == selected_product)
    ]
    # Calculate min and avg farm price within the selected date range
    min_farm_price = filtered_farm_data['Unit Price'].min()
    avg_farm_price = filtered_farm_data['Unit Price'].mean()
    
    
    
    # Loop over each row in the farm data to compare
    for index, row in filtered_farm_data.iterrows():
        date_ = row['Timestamp']  # Use the actual date from farm_data
        location = row['Location']
        farm_price = row['Unit Price']
        
        supplier_name = row.get('Supplier Name', 'Unknown Suppler')
        product_name = row.get('Products List', 'Unknown Product')
        
        # Ensure farm_price is a float
        try:
            farm_price = float(farm_price)
        except ValueError:
            st.write(f"Invalid farm price found at index {index} - Value: {farm_price}")
            st.write(f"Row details: {row}")
            continue  # Skip this row if farm_price is not a valid number
        
        
        # Check if there are corresponding benchmark prices
        for group, prices in benchmark_prices.items():
            min_price = prices.get("min_price", None)
            avg_price = prices.get("avg_price", None)
            
            # Ensure min_price and avg_price are numbers (not None or NaN)
            if min_price is not None and not math.isnan(min_price) and avg_price is not None and not math.isnan(avg_price):
                min_diff = ((min_farm_price - min_price) / min_price) * 100
                avg_diff = ((avg_farm_price - avg_price) / avg_price) * 100
                
                # Store comparison results
                comparison_results.append({
                    "date_": date_,
                    "Products List": product_name,
                    "Supplier Name": supplier_name,
                    "Location": location,
                    # "Farm Price (Unit Price)": farm_price,
                    "Min Farm Price": min_farm_price,
                    "Avg Farm Price": avg_farm_price,
                    f"Min Price ({group})": min_price,
                    f"Min Difference % ({group})": min_diff,
                    f"Avg Price ({group})": avg_price,
                    f"Avg Difference % ({group})": avg_diff
                })
    
    # Convert to DataFrame for easier display
    return pd.DataFrame(comparison_results)

def get_min_avg_prices_by_location_and_date(farm_data, selected_date_range, selected_product):
    start_date, end_date = selected_date_range
    filtered_farm_data = farm_data[
        (farm_data['Timestamp'] >= start_date) &
        (farm_data['Timestamp'] <= end_date) &
        (farm_data['Products List'] == selected_product)
    ]

    # Group by Location, Date, and Product, then calculate the minimum and average price for each group
    location_date_prices = (
        filtered_farm_data.groupby(["Location", "Timestamp", "Products List"])["Unit Price"]
        .agg(Min_Farm_Price='min', Avg_Farm_Price='mean')
        .reset_index()
    )
    # location_date_prices.columns = ["Location", "Timestamp", "Products List","Min Farm Price","Avg Farm Price"]
    return location_date_prices


def compare_farm_prices_with_benchmarks_by_location(farm_data, benchmark_prices, selected_date_range):
    comparison_results = []
    min_prices_by_location_and_date = get_min_avg_prices_by_location_and_date(farm_data, selected_date_range, selected_product)
    # st.write(min_prices_by_location_and_date)
    for _, row in min_prices_by_location_and_date.iterrows():
        date_ = row['Timestamp']
        location = row['Location']
        min_farm_price = row['Min_Farm_Price']
        avg_farm_price = row['Avg_Farm_Price']
        product_name = row.get('Products List', 'Unknown Product')  # Retrieve the product name
        
        # Compare with benchmark prices for each location group
        for group, prices in benchmark_prices.items():
            min_price = prices.get("min_price", None)
            avg_price = prices.get("avg_price", None)
            # Ensure min_price and avg_price are numbers
            if min_price is not None:
                try:
                    min_price = float(min_price)
                except ValueError:
                    min_price = None  # Set to None if conversion fails

            if avg_price is not None:
                try:
                    avg_price = float(avg_price)
                except ValueError:
                    avg_price = None  # Set to None if conversion fails

            
            if min_price is not None and not math.isnan(min_price):
                min_diff = ((min_farm_price - min_price) / min_price) * 100
            else:
                min_diff = None
            
            if avg_price is not None and not math.isnan(avg_price):
                avg_diff = ((avg_farm_price - avg_price) / avg_price) * 100
            else:
                avg_diff = None
            
            # Store comparison results
            comparison_results.append({
                "date_": date_,
                "Products List": product_name,
                "Location": location,
                "Min Farm Price": min_farm_price,
                "Avg Farm Price": avg_farm_price,
                f"Min Price ({group})": min_price,
                f"Min Difference % ({group})": min_diff,
                f"Avg Price ({group})": avg_price,
                f"Avg Difference % ({group})": avg_diff
            })
    
    # Convert to DataFrame for easier display
    return pd.DataFrame(comparison_results)



# Use the function
farm_data_filtered = survey_3[survey_3['Products List'] == selected_product]
# st.write(farm_data_filtered)
benchmark_prices = calculate_benchmark_prices(filtered_survey, selected_date_range, selected_product)
price_comparison_df = compare_farm_prices_with_benchmarks(farm_data_filtered, benchmark_prices,selected_date_range)

# st.write(price_comparison_df)
min_prices_by_location_and_date = get_min_avg_prices_by_location_and_date(farm_data_filtered, selected_date_range, selected_product)
# st.write(min_prices_by_location_and_date)
# min_prices_by_location = get_min_prices_by_location(farm_data_filtered, selected_date_range, selected_product)
comparison_results = compare_farm_prices_with_benchmarks_by_location(farm_data_filtered, benchmark_prices,  selected_date_range)

# KPIs - Calculate performance metrics
st.subheader("Performance KPIs")

# Create a function to filter the data based on selected product and date range
def filter_data_by_product_and_date(price_comparison_df, selected_product,selected_date_range):
    price_comparison_df['date_'] = pd.to_datetime(price_comparison_df['date_'])
    
    # Convert start_date and end_date to datetime objects for consistency
  
    start_date, end_date = selected_date_range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_df = price_comparison_df[
        (price_comparison_df['Products List'] == selected_product) &
        (price_comparison_df['date_'] >= start_date) &
        (price_comparison_df['date_'] <= end_date)
    ]
    return filtered_df

def aggregate_price_difference_by_frequency(df, frequency):
    # Ensure the date column is in datetime format
    df['date_'] = pd.to_datetime(df['date_'])
    
    # Define all possible columns for aggregation
    min_columns = ['Min Farm Price','Min Difference % (Local Shops)', 'Min Difference % (Supermarkets)', 
                   'Min Difference % (Sunday Markets)', 'Min Difference % (Distribution Centers)']
    
    avg_columns = ['Avg Farm Price','Avg Difference % (Local Shops)', 'Avg Difference % (Supermarkets)', 
                   'Avg Difference % (Sunday Markets)', 'Avg Difference % (Distribution Centers)']
    
    # Find which columns exist in the DataFrame
    existing_min_columns = [col for col in min_columns if col in df.columns]
    existing_avg_columns = [col for col in avg_columns if col in df.columns]

    # Create a dictionary for aggregation based on available columns
    agg_dict = {}
    
    for col in existing_min_columns:
        agg_dict[col] = 'min'
    
    for col in existing_avg_columns:
        agg_dict[col] = 'mean'

    # Group by Supplier Name and selected frequency only for the existing columns
    if agg_dict:
        grouped_df = df.groupby([
            'Supplier Name',
            pd.Grouper(key='date_', freq=frequency)
        ]).agg(agg_dict).reset_index()
    else:
        # If no relevant columns exist, return an empty DataFrame
        grouped_df = pd.DataFrame()

    return grouped_df

def aggregate_price_difference_by_frequency_location(df, frequency):
    # Ensure the date column is in datetime format
    df['date_'] = pd.to_datetime(df['date_'])
    
    # Define all possible columns for aggregation
    min_columns = ['Min Farm Price','Min Difference % (Local Shops)', 'Min Difference % (Supermarkets)', 
                   'Min Difference % (Sunday Markets)', 'Min Difference % (Distribution Centers)']
    
    avg_columns = ['Avg Farm Price','Avg Difference % (Local Shops)', 'Avg Difference % (Supermarkets)', 
                   'Avg Difference % (Sunday Markets)', 'Avg Difference % (Distribution Centers)']
    
    # Find which columns exist in the DataFrame
    existing_min_columns = [col for col in min_columns if col in df.columns]
    existing_avg_columns = [col for col in avg_columns if col in df.columns]

    # Create a dictionary for aggregation based on available columns
    agg_dict = {}
    
    for col in existing_min_columns:
        agg_dict[col] = 'min'
    
    for col in existing_avg_columns:
        agg_dict[col] = 'mean'

    # Group by Supplier Name and selected frequency only for the existing columns
    if agg_dict:
        grouped_df = df.groupby([
            'Location',
            pd.Grouper(key='date_', freq=frequency)
        ]).agg(agg_dict).reset_index()
    else:
        # If no relevant columns exist, return an empty DataFrame
        grouped_df = pd.DataFrame()

    return grouped_df
# Apply the filtering
filtered_price_comparison_df = filter_data_by_product_and_date(price_comparison_df, selected_product, selected_date_range)
filtered_price_comparision_by_location = filter_data_by_product_and_date(comparison_results, selected_product, selected_date_range)
# st.dataframe(filtered_price_comparision_by_location)
import streamlit as st
import pandas as pd
import plotly.express as px
# Use the selected frequency
# Add Frequency Selection in Streamlit
frequency_options = {
    'Daily': 'D',
    'Weekly': 'W',
    'Monthly': 'M',
}
selected_frequency = st.sidebar.selectbox('Select Frequency', list(frequency_options.keys()))

price_aggregated = aggregate_price_difference_by_frequency(filtered_price_comparison_df, frequency_options[selected_frequency])
price_by_location_aggregated = aggregate_price_difference_by_frequency_location(filtered_price_comparision_by_location, frequency_options[selected_frequency])
try:
    if not price_comparison_df.empty:
        st.write("Price Comparison between Farm Buying Prices and Benchmark Prices based on Vendors")
        st.dataframe(price_aggregated)
    else:
        st.warning("No price comparison data found for the selected criteria.")
except Exception as e:
    st.error(f"Failed to display price comparison: {e}")
try:
    if not price_by_location_aggregated.empty:
        st.write("Price Comparison between Farm Buying Prices and Benchmark Prices based on Location")
        st.dataframe(price_by_location_aggregated)
    else:
        st.warning("No price comparison data found for the selected criteria.")
except Exception as e:
    st.error(f"Failed to display price comparison: {e}")
    

def visualize_min_avg_comparison(price_aggregated, selected_frequency, x_column, xaxis_title, chart_key_suffix):
    # List of possible columns for Min and Avg price differences
    min_columns = [
        'Min Difference % (Local Shops)', 
        'Min Difference % (Supermarkets)', 
        'Min Difference % (Sunday Markets)', 
        'Min Difference % (Distribution Centers)'
    ]

    avg_columns = [
        'Avg Difference % (Local Shops)', 
        'Avg Difference % (Supermarkets)', 
        'Avg Difference % (Sunday Markets)', 
        'Avg Difference % (Distribution Centers)'
    ]

    # Filter out columns that are not present in the DataFrame
    available_min_columns = [col for col in min_columns if col in price_aggregated.columns]
    available_avg_columns = [col for col in avg_columns if col in price_aggregated.columns]

    # If there are no available columns, warn the user and exit
    if not available_min_columns:
        st.warning("No data available for Min price comparison.")
        return
    if not available_avg_columns:
        st.warning("No data available for Avg price comparison.")
        return

    # Create Min price comparison chart
    fig_min = px.bar(
        price_aggregated.dropna(subset=available_min_columns), 
        x=x_column, 
        y=available_min_columns,
        color_discrete_sequence=['#0055A4', '#6CA0DC', '#FF4D4D', '#FFC1C1'],
        barmode='group',
        title=f"Min Price Comparison Against Benchmark Markets - {selected_frequency} Aggregation",
        labels={'value': 'Price Difference (%)', 'variable': 'Benchmark Market'},
        hover_data={'date_': True}
    )

    fig_min.update_layout(
        title_x=0.5,
        yaxis_title='Price Difference (%)',
        xaxis_title=xaxis_title,
        legend_title_text='Benchmark Market',
        xaxis_tickangle=-45
    )
    
    # Create Avg price comparison chart
    fig_avg = px.bar(
        price_aggregated.dropna(subset=available_avg_columns), 
        x=x_column, 
        y=available_avg_columns,
        color_discrete_sequence=['#0055A4', '#6CA0DC', '#FF4D4D', '#FFC1C1'],
        barmode='group',
        title=f"Avg Price Comparison Against Benchmark Markets - {selected_frequency} Aggregation",
        labels={'value': 'Price Difference (%)', 'variable': 'Benchmark Market'},
        hover_data={'date_': True}
    )

    fig_avg.update_layout(
        title_x=0.5,
        yaxis_title='Price Difference (%)',
        xaxis_title=xaxis_title,
        legend_title_text='Benchmark Market',
        xaxis_tickangle=-45
    )

    # Display charts in Streamlit with unique keys
    st.plotly_chart(fig_min, use_container_width=True, key=f"min_chart_{chart_key_suffix}")
    st.plotly_chart(fig_avg, use_container_width=True, key=f"avg_chart_{chart_key_suffix}")

# Call this function with different parameters based on the DataFrame being used
try:
    if not price_aggregated.empty:
        st.write(f"Visualizing Price Comparison for {selected_product} Between Vendors and Benchmark Markets")
        visualize_min_avg_comparison(price_aggregated, selected_frequency, x_column='Supplier Name', xaxis_title='Supplier Name', chart_key_suffix='price_aggregated')
    else:
        st.warning("No price comparison data found for the selected product and date range.")
except Exception as e:
    st.error(f"Failed to display price comparison visualization: {e}")

try:
    if not price_by_location_aggregated.empty:
        st.write(f"Visualizing Price Comparison for {selected_product} Between Vendors and Benchmark Markets")
        visualize_min_avg_comparison(price_by_location_aggregated, selected_frequency, x_column='Location', xaxis_title='Location', chart_key_suffix='price_by_location_aggregated')
    else:
        st.warning("No price comparison data found for the selected product and date range.")
except Exception as e:
    st.error(f"Failed to display price comparison visualization: {e}")
import plotly.express as px
import plotly.graph_objs as go

def plot_price_trend(farm_data, selected_product, selected_vendor, selected_date_range, frequency): 
    # Unpack start and end dates from the selected range
    start_date, end_date = [pd.to_datetime(date) for date in selected_date_range]
    pandas_frequency = frequency_options.get(frequency)
    
    # Ensure valid frequency
    if pandas_frequency is None:
        st.error(f"Invalid frequency: {frequency}")
        return

    # Convert 'date_' column to datetime format
    farm_data['date_'] = pd.to_datetime(farm_data['date_'])
    
    # Filter data based on the selected product, vendor, and date range
    filtered_data = farm_data[
        (farm_data['Products List'] == selected_product) &
        (farm_data['Supplier Name'] == selected_vendor) &
        (farm_data['date_'] >= start_date) &
        (farm_data['date_'] <= end_date)
    ]
    
    # If a frequency is provided, resample the data accordingly
    if frequency:
        resampled_data = (
            filtered_data.set_index('date_')
            .resample(pandas_frequency)['Min Difference % (Local Shops)', 'Min Difference % (Supermarkets)', 'Min Difference % (Sunday Markets)']
            .mean()
            .reset_index()
        )
    else:
        resampled_data = filtered_data

    # Format date for better readability (only date, no time)
    resampled_data['date_'] = resampled_data['date_'].dt.date
    full_date_range = pd.date_range(
        start_date, 
        end_date,
        freq='D'
    )
    # Create the Plotly figure
    fig = go.Figure()

    # Add traces for each vendor type with data labels
    fig.add_trace(go.Scatter(
        x=resampled_data['date_'], 
        y=resampled_data['Min Difference % (Local Shops)'], 
        mode='lines+markers+text', 
        name='Min Difference % (Local Shops)',
        line=dict(color='blue'),
        text=resampled_data['Min Difference % (Local Shops)'].round(2),  # Show values rounded to 2 decimals
        textposition="top center"
    ))

    fig.add_trace(go.Scatter(
        x=resampled_data['date_'], 
        y=resampled_data['Min Difference % (Supermarkets)'], 
        mode='lines+markers+text', 
        name='Min Difference % (Supermarkets)',
        line=dict(color='red'),
        text=resampled_data['Min Difference % (Supermarkets)'].round(2),
        textposition="top center"
    ))

    fig.add_trace(go.Scatter(
        x=resampled_data['date_'], 
        y=resampled_data['Min Difference % (Sunday Markets)'], 
        mode='lines+markers+text', 
        name='Min Difference % (Sunday Markets)',
        line=dict(color='orange'),
        text=resampled_data['Min Difference % (Sunday Markets)'].round(2),
        textposition="top center"
    ))

    # Set title and axis labels
    fig.update_layout(
        title=f"Min Difference % Comparison for {selected_product} ({selected_vendor})",
        xaxis_title="Date",
        yaxis_title="Min Difference %",
        legend_title="Vendor Types",
        template="plotly_white",
        xaxis=dict(
        tickmode='array',  # Define specific tick values
        tickvals=full_date_range,  # Specify the tick positions (dates)
        ticktext=[d.strftime('%Y-%m-%d') for d in full_date_range],  # Format tick labels
        tickformat="%Y-%m-%d",  # Ensure date formatting
        tickangle=-45
    )
    )

    # Display the interactive Plotly plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

selected_vendor = st.sidebar.selectbox('Select Vendor', price_comparison_df['Supplier Name'].unique())

# Call the function to plot
plot_price_trend(price_comparison_df, selected_product, selected_vendor, selected_date_range, selected_frequency)
