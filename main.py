from modules.lib import *
import config
from database import *

# Streamlit app configuration
st.set_page_config(page_title="Satellite Imagery & NDVI Dashboard", layout="wide")
st.markdown(
    """
    <h1 style='text-align: cent er; 
               font-size: 40px; 
               color: #FF5733; 
               background-color: #a6a6a6; 
               padding: 10px; 
               border-radius: 10px; 
               border-bottom: 3px solid #4CAF50;'>
        🌍 NDVI Dashboard 🌿
    </h1>
    <br>
    <br>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #a6a6a6;
            border-right: 4px solid #4CAF50;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #4CAF50;'>🌿 NDVI Analyzer</h2>",
    unsafe_allow_html=True
)

# Sidebar for user input
st.sidebar.header("INPUT PARAMETERS 📍")

# Check if credentials are properly set
if not config.config.sh_client_id or not config.config.sh_client_secret:
    st.error("Sentinel Hub Client ID and Client Secret are not set. Please check your config.py file.")
else:
    st.toast("Sentinel Hub credentials are set correctly!", icon="✅")


# Initialize session state for location if not already initialized
if 'location_input' not in st.session_state:
    st.session_state.location_input = "New Delhi"

def save_image_as_png(image_array):
    """
    Convert a NumPy image array to a PNG BytesIO object.
    """
    # Ensure the image is in 8-bit format
    image_8bit = (image_array * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_8bit)

    # Save as PNG into BytesIO
    image_bytes = io.BytesIO()
    image_pil.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    
    return image_bytes

# Function to get coordinates from Nominatim
def get_coordinates(location_name):
    url = f"https://nominatim.openstreetmap.org/search?q={location_name}&format=json&limit=1"
    response = requests.get(url, headers={"User-Agent": "Streamlit App"})
    if response.status_code == 200:
        results = response.json()
        if results:
            coords = results[0]
            return float(coords["lat"]), float(coords["lon"])
        else:
            st.error("Location not found.")
            return None, None
    else:
        st.error("Failed to connect to the Nominatim API.")
        return None, None


# Safe division function for vegetation index calculations
def safe_divide(numerator, denominator):
    denominator[denominator == 0] = np.nan  # Replace zeros with NaN
    return numerator / denominator

# Functions to calculate vegetation indices
def calculate_ndvi(nir_band, red_band):
    denominator = nir_band + red_band
    return safe_divide(nir_band - red_band, denominator)

def calculate_evi(nir_band, red_band, green_band):
    denominator = nir_band + 6.0 * red_band - 7.5 * green_band + 1
    return safe_divide(2.5 * (nir_band - red_band), denominator)

def calculate_savi(nir_band, red_band):
    denominator = nir_band + red_band + 0.5
    return safe_divide(nir_band - red_band, denominator) * 1.5


# Function to save Graph data to the database
def save_graph_to_db(report_name, location, data):
    conn = sqlite3.connect("NDVI_reports_.db")
    cursor = conn.cursor()
    
    # Insert each row into the graph_reports table
    for _, row in data.iterrows():
        cursor.execute(
            "INSERT INTO Graph_reports (report_name, location, date, ndvi_value) VALUES (?, ?, ?, ?)", 
            (report_name, location, row["Date"], row["NDVI Value"])
        )

    conn.commit()
    conn.close()
    st.success(f"Graph Report '{report_name}' has been saved to the database.")

# Function to save histogram data to the database
def save_histogram_to_db(report_name, location, data, report_type="Histogram"):
    conn = sqlite3.connect("NDVI_reports_.db")
    cursor = conn.cursor()

    # Convert DataFrame to CSV string
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # Insert into the histogram_reports table
    cursor.execute("INSERT INTO histogram_reports (report_name, location, report_type, data) VALUES (?, ?, ?, ?)", 
                   (report_name, location, report_type, csv_content))
    
    conn.commit()
    conn.close()
    st.success(f"Histogram Report '{report_name}' has been saved to the database.")

def create_download_button(data, filename, button_label):
    # Convert DataFrame to CSV format
    df = pd.DataFrame(data)
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # Create download button
    st.sidebar.download_button(
        label=button_label,
        data=csv_content,
        file_name=filename,
        mime="text/csv"
    )

# Determine initial coordinates
latitude, longitude = get_coordinates(st.session_state.location_input)
selected_coordinates = [latitude, longitude] if latitude and longitude else [28.6139, 77.2090]  # Default: New Delhi

# Location search input
with st.sidebar.expander("📍 Location"):
    location_input = st.text_input("Enter Location", st.session_state.location_input)

# User inputs for date range
with st.sidebar.expander("📅 Select Date Range"):
    start_date = st.date_input("Start Date", value=datetime.date(2023, 1, 1))
    end_date = st.date_input("End Date", value=datetime.date(2023, 1, 15))

# Update the map if the location input changes
if location_input != st.session_state.location_input:
    new_latitude, new_longitude = get_coordinates(location_input)
    if new_latitude and new_longitude:
        selected_coordinates = [new_latitude, new_longitude]
        st.session_state.location_input = location_input
        st.rerun()  # Trigger a rerun to refresh the map

st.subheader("📍 INTERACTIVE LOCATION SELECTION")

# Interactive map for user location selection with measure control and mouse position
folium_map = folium.Map(location=selected_coordinates, zoom_start=6)
marker = folium.Marker(selected_coordinates, popup="Selected Location", draggable=True)
marker.add_to(folium_map)
folium_map.add_child(MeasureControl())
folium_map.add_child(MousePosition())

map_data = st_folium(folium_map, width=1000, height=600, key="location_map")

# Update coordinates when the user clicks on the map
if map_data and map_data.get("last_clicked") is not None:
    new_lat, new_lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]

    # Reverse geocode to get the location name
    reverse_geocode_url = f"https://nominatim.openstreetmap.org/reverse?lat={new_lat}&lon={new_lon}&format=json"
    response = requests.get(reverse_geocode_url, headers={"User-Agent": "Streamlit App"})

    if response.status_code == 200:
        location_info = response.json()

        # Extract city, town, or village
        new_location = (
            location_info.get("address", {}).get("city") or
            location_info.get("address", {}).get("town") or
            location_info.get("address", {}).get("village") or
            location_info.get("display_name", "Unknown Location")
        )

        # Only update the session state if the location changed
        if new_location != st.session_state.get("location_input"):
            st.session_state.location_input = new_location
            st.session_state.selected_coordinates = [new_lat, new_lon]
            st.rerun()  # Force Streamlit to refresh with new location



# Define bounding box and request satellite image
bbox = BBox((selected_coordinates[1] - 0.01, selected_coordinates[0] - 0.01, selected_coordinates[1] + 0.01, selected_coordinates[0] + 0.01), CRS.WGS84)
evalscript = """
// Script to fetch Red, NIR for NDVI, and True Color
function setup() {
    return {
        input: ["B02", "B03", "B04", "B08"],
        output: { bands: 4 }
    };
}

function evaluatePixel(sample) {
    return [sample.B04, sample.B03, sample.B02, sample.B08]; // Red, Green, Blue, NIR
}
"""
request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[SentinelHubRequest.input_data(
        data_collection=DataCollection.SENTINEL2_L2A,
        time_interval=(str(start_date), str(end_date)),
        mosaicking_order="leastCC"
    )],
    responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
    bbox=bbox, 
    size=(1024, 1024),
    config=config.config
)


# Process satellite imagery
def process_image():
    try:
        image = request.get_data()[0]
        if image.shape[2] != 4:
            st.error("Retrieved image does not contain expected bands.")
            return None

        # Normalize bands
        red_band = image[:, :, 0].astype(float) / 255.0
        green_band = image[:, :, 1].astype(float) / 255.0
        blue_band = image[:, :, 2].astype(float) / 255.0
        nir_band = image[:, :, 3].astype(float) / 255.0

        true_color = np.stack([red_band, green_band, blue_band], axis=-1)
        infrared_image = np.stack([nir_band, red_band, green_band], axis=-1)

        return true_color, infrared_image, red_band, green_band, nir_band

    except Exception as e:
        st.error(f"Error retrieving satellite image: {e}")
        return None
    
    
# Display selected vegetation index
if selected_coordinates[0] and selected_coordinates[1]:
    st.write("## Satellite Imagery & Vegetation Indices")
    data = process_image()

    image_descriptions = {
        "True Color": {
            "short": "Represents natural colors as seen by the human eye.",
        },
        "Infrared": {
            "short": "Highlights vegetation using near-infrared light.",
        },
        "NDVI": {
            "short": "Measures vegetation health using red and NIR bands.",
        },
        "EVI": {
            "short": "Enhanced NDVI that reduces atmospheric distortions.",
        },
        "SAVI": {
            "short": "Modified NDVI for arid regions with sparse vegetation.",
        }
    }

    if data:
        true_color, infrared_image, red_band, green_band, nir_band = data

        # Dropdowns for user selections
        with st.sidebar.expander("🖼️ Select Image Type"):
            image_type = st.selectbox("Categories:", ["True Color", "Infrared"])
        with st.sidebar.expander("🌿 Vegetation Index Selection"):
            index = st.selectbox("Indexes:", ["NDVI", "EVI", "SAVI"])

        if image_type == "True Color":
            # Image and download button placed tightly
            # img caption
            st.image(true_color, use_container_width=True)
            st.markdown(f"<p style='text-align:center; font-size:23px; font-weight:bold;'>{image_type} Image</p>", unsafe_allow_html=True)
            
            # informatin text box
            st.markdown(f"<p style='font-size:21px;'><b>{image_type}: {image_descriptions[image_type]['short']}</b></p>", unsafe_allow_html=True)


            # Download button - True Color Image
            st.download_button(
                label="Download True Color Image",
                # data=BytesIO(Image.fromarray((true_color * 255).astype(np.uint8)).tobytes()),
                data =save_image_as_png(true_color),
                file_name="true_color_image.png",
                mime="image/png"
            )

        else:
            st.image(infrared_image, use_container_width=True)
            # image caption
            st.markdown(
                f"<p style='text-align:center; font-size:23px; font-weight:bold;'>{image_type} Image</p>", 
                unsafe_allow_html=True)
            # information text box
            st.markdown(f"<p style='font-size:21px;'><b>{image_type}: {image_descriptions[image_type]['short']}</b></p>", unsafe_allow_html=True)
            # Download button - infra Color Image
            st.download_button(
                label="Download infrared Color Image",
                # data=BytesIO(Image.fromarray((infrared_image * 255).astype(np.uint8)).tobytes()),
                data =save_image_as_png(infrared_image),
                file_name="infrared_color_image.png",
                mime="image/png"
            )
        st.markdown("<hr style='border: 1px solid #f0ffff; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)


        if index == "NDVI":
            vegetation_index = calculate_ndvi(nir_band, red_band)
        elif index == "EVI":
            vegetation_index = calculate_evi(nir_band, red_band, green_band)
        else:
            vegetation_index = calculate_savi(nir_band, red_band)

        vegetation_index = np.clip(vegetation_index, -1, 1)
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        vegetation_index_color = cmap(norm(vegetation_index))

        # Normalize and convert vegetation index image to PNG format
        vegetation_index_8bit = (vegetation_index_color[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel
        vegetation_pil = Image.fromarray(vegetation_index_8bit)

        #Save as PNG into BytesIO object
        vegetation_bytes = io.BytesIO()
        vegetation_pil.save(vegetation_bytes, format="PNG")
        vegetation_bytes.seek(0)  # Move pointer to start

        # Display NDVI/EVI/SAVI Image
        st.image(vegetation_index_color, use_container_width=True)
        #image caption
        st.markdown(
            f"<p style='text-align:center; font-size:23px; font-weight:bold;'>{index} Image</p>", 
            unsafe_allow_html=True
        )
        # info textbox
        st.markdown(f"<p style='font-size:21px;'><b>{index}: {image_descriptions[index]['short']}</b></p>", unsafe_allow_html=True)

        # Add download button directly below NDVI/EVI/SAVI Image
        st.download_button(
            label=f"Download {index} Image",
            # data=BytesIO(Image.fromarray((vegetation_index_color * 255).astype(np.uint8)).tobytes()),
            data=vegetation_bytes,
            file_name=f"{index.lower()}_image.png",
            mime="image/png"
        )
        st.markdown("<hr style='border: 1px solid #f0ffff; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)


        # Vegetation Index Graph
        # Generate Date Range
        dates = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        actual_ndvi_values = calculate_ndvi(nir_band, red_band).flatten()[:len(dates)]
        
        st.markdown("<h3 style='text-align: left; color: #b2beb5;'>Index Graph</h3>", unsafe_allow_html=True)
        
        # Create Plotly Interactive Line Graph
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates, y=actual_ndvi_values, 
            mode='lines+markers', 
            line=dict(color='green', width=2),
            marker=dict(size=8, color='red'),
            name=f"{index} Values"
        ))

        # Enhance Layout
        fig.update_layout(
            title=dict(
                text=f"🌿 {index} Trend for {st.session_state.location_input}",
                font=dict(size=18, color="#4CAF50"),
                x=0.4  # Center Align Title
            ),
            xaxis_title="Date",
            yaxis_title=f"{index} Value",
            xaxis=dict(
                tickformat="%d/%m",
                tickangle=-45,
                showgrid=True,
                gridwidth=0.5,
                gridcolor="lightgrey"
            ),
            yaxis=dict(showgrid=True, gridcolor="lightgrey"),
            legend=dict(font=dict(size=12)),
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent Background
        )

        # Display in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Graph DB Saving Button
        if st.button(f"Save {index} Graph Data to Database"):
            save_graph_to_db(
                f"{index}_Graph_{st.session_state.location_input}", 
                st.session_state.location_input, 
                pd.DataFrame({'Date': dates, 'NDVI Value': actual_ndvi_values})
            )

        # Generate CSV 
        graph_csv = pd.DataFrame({'Date': dates, f'{index}': actual_ndvi_values}).to_csv(index=False)
        # Add the CSV download button for graph data directly below the graph
        st.download_button(
            label=f"Download {index} Graph Data as CSV",
            data=graph_csv,
            file_name=f"{index.lower()}_graph_data.csv",
            mime="text/csv"
        )
        st.markdown("<hr style='border: 1px solid #f0ffff; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)


        # HISTOGRAM
        st.markdown("<h3 style='text-align: left; color: #b2beb5;'>NDVI Value Distribution</h3>", unsafe_allow_html=True)

        # Frequency distribution histogram
        freq_index = vegetation_index.flatten()
        freq_index = freq_index[~np.isnan(freq_index)]  # Remove NaN values

        # Convert data to a Pandas DataFrame
        df = pd.DataFrame({'Value': freq_index})

        # Create an interactive histogram using Plotly
        fig = px.histogram(df, x='Value', nbins=50, color_discrete_sequence=['green'])
        fig.update_layout(
            title=dict(
                text=f"🌿 Frequency Distribution of {index}",
                font=dict(size=18, color="#4CAF50"),
                x=0.4  # Center Align Title
            ),
            xaxis_title="NDVI Value",
            yaxis_title="Frequency",
            bargap=0.1
        )

        # Display Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

         # Histogram DB Saving Button
        if st.button(f"Save {index} Histogram Data to Database"):
            save_histogram_to_db(f"{index}_Histogram_{st.session_state.location_input}", 
                                st.session_state.location_input, 
                                pd.DataFrame({'Value': freq_index}), 
                                "Histogram")
            
        # Generate CSV
        histogram_csv = df.to_csv(index=False)

        # Add CSV download button for histogram data
        st.download_button(
            label=f"Download {index} Histogram Data as CSV",
            data=histogram_csv,
            file_name=f"{index.lower()}_histogram_data.csv",
            mime="text/csv"
        )


st.sidebar.markdown("---")  # Adds a separator
st.sidebar.markdown(
    "<p style='text-align: center; font-size: 12px; color: black;'>Developed by <b>A11</b> | Data from Sentinel Hub</p>",
    unsafe_allow_html=True
)