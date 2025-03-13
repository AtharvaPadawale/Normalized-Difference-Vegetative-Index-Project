from modules.lib import *
import config
from database import *

# Streamlit app configuration
st.set_page_config(page_title="Satellite Imagery & NDVI Dashboard", layout="wide")
st.title("🌍 NDVI Dashboard")

# Check if credentials are properly set
if not config.config.sh_client_id or not config.config.sh_client_secret:
    st.error("Sentinel Hub Client ID and Client Secret are not set. Please check your config.py file.")
else:
    st.toast("Sentinel Hub credentials are set correctly!", icon="✅")

# Sidebar for user input
st.sidebar.header("INPUT PARAMETERS 📍")
st.sidebar.subheader("Location and Date Range")

# Initialize session state for location if not already initialized
if 'location_input' not in st.session_state:
    st.session_state.location_input = "New Delhi"

def save_image_as_png(image_array):
    """
    Convert a NumPy image array (0-1 float or 0-255 uint8) to a PNG BytesIO object.
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

# save CSV reports to the Database 
def save_csv_to_db(data, report_name, report_type):
    df = pd.DataFrame(data)
    save_report_to_db(report_name, df, report_type)
    st.success(f"Report '{report_name}' has been saved to the database.")

def create_download_button(data, filename, button_label):
    csv = save_csv_to_db(data, filename)
    st.sidebar.download_button(
        label=button_label,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# Determine initial coordinates
latitude, longitude = get_coordinates(st.session_state.location_input)
selected_coordinates = [latitude, longitude] if latitude and longitude else [28.6139, 77.2090]  # Default: New Delhi

# Location search input
location_input = st.sidebar.text_input("Enter Location (e.g., 'Mumbai', 'Delhi')", st.session_state.location_input)

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

# After the user clicks on the map and selects a location
if map_data and map_data.get("last_clicked") is not None:
    new_lat, new_lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]

    # Reverse geocode to get location name
    reverse_geocode_url = f"https://nominatim.openstreetmap.org/reverse?lat={new_lat}&lon={new_lon}&format=json"
    response = requests.get(reverse_geocode_url)

    if response.status_code == 200:
        location_info = response.json()
        new_location = (
            location_info["address"].get("city") or
            location_info["address"].get("town") or
            location_info["address"].get("village") or
            "Unknown Location"
        )

        # Only update session state if the location changed
        if new_location and new_location != st.session_state.get("location_input"):
            st.session_state.location_input = new_location
            st.session_state.selected_coordinates = [new_lat, new_lon]  # Update stored coordinates
            st.rerun()  # Force Streamlit to refresh

# User inputs for date range
start_date = st.sidebar.date_input("Start Date", value=datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.date(2023, 1, 15))

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
        time_interval=(str(start_date), str(end_date))
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
    if data:
        true_color, infrared_image, red_band, green_band, nir_band = data

        # Dropdowns for user selections
        image_type = st.sidebar.selectbox("Select Image Type", ["True Color", "Infrared"])
        index = st.sidebar.selectbox("Select Vegetation Index", ["NDVI", "EVI", "SAVI"])

        if image_type == "True Color":
            # Image and download button placed tightly
            st.image(true_color, caption="True Color Image", use_container_width=True)
            # Download button - True Color Image
            st.download_button(
                label="Download True Color Image",
                # data=BytesIO(Image.fromarray((true_color * 255).astype(np.uint8)).tobytes()),
                data =save_image_as_png(true_color),
                file_name="true_color_image.png",
                mime="image/png"
            )
        else:
            st.image(infrared_image, caption="Infrared Image", use_container_width=True)
            # Download button - infra Color Image
            st.download_button(
                label="Download infrared Color Image",
                # data=BytesIO(Image.fromarray((infrared_image * 255).astype(np.uint8)).tobytes()),
                data =save_image_as_png(infrared_image),
                file_name="infrared_color_image.png",
                mime="image/png"
            )

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
        st.image(vegetation_index_color, caption=f"{index} Image", use_container_width=True)

        # Add download button directly below NDVI/EVI/SAVI Image
        st.download_button(
            label=f"Download {index} Image",
            # data=BytesIO(Image.fromarray((vegetation_index_color * 255).astype(np.uint8)).tobytes()),
            data=vegetation_bytes,
            file_name=f"{index.lower()}_image.png",
            mime="image/png"
        )

        # Vegetation Index Graph
        dates = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        actual_ndvi_values = calculate_ndvi(nir_band, red_band).flatten()[:len(dates)]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, actual_ndvi_values, label=f"{index} Values", color="green")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title(f"{index} for {st.session_state.location_input}")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Save to database when button is clicked
        if st.button(f"Save {index} Graph Data to Database"):
            save_csv_to_db({'Date': dates, f'{index}': actual_ndvi_values}, f"{index}_Graph_{st.session_state.location_input}", "Graph")

        # Generate CSV 
        graph_csv = pd.DataFrame({'Date': dates, f'{index}': actual_ndvi_values}).to_csv(index=False)
        # Add the download button for graph data directly below the graph
        st.download_button(
            label=f"Download {index} Graph Data as CSV",
            data=graph_csv,
            file_name=f"{index.lower()}_graph_data.csv",
            mime="text/csv"
        )

        # Frequency distribution histogram
        freq_index = vegetation_index.flatten()
        freq_index = freq_index[~np.isnan(freq_index)]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(freq_index, bins=20, color="blue", edgecolor="black", alpha=0.7)
        ax.set_title(f"Frequency Distribution of {index}")
        st.pyplot(fig)

        # Save histogram data to database when button is clicked
        if st.button(f"Save {index} Histogram Data to Database"):
            save_csv_to_db({'Value': freq_index}, f"{index}_Histogram_{st.session_state.location_input}", "Histogram")

        # Generate CSV 
        histogram_csv = pd.DataFrame({'Value': freq_index}).to_csv(index=False)
        # Add the download button directly below the histogram
        st.download_button(
            label=f"Download {index} Histogram Data as CSV",
            data=histogram_csv,
            file_name=f"{index.lower()}_histogram_data.csv",
            mime="text/csv"
        )

