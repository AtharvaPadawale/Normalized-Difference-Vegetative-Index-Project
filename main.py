from modules.lib import *
import config
from database import *

# Streamlit app configuration
st.set_page_config(page_title="NDVI TerraMetrics Dashboard", layout="wide")

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');
        
        .glass-title {
            text-align: center; 
            font-size: 38px; 
            font-family: 'Orbitron', sans-serif;
            color: #D4E7F5; 
            padding: 15px;
            border-radius: 10px;
            background: rgba(20, 20, 20, 0.5);
            /* box-shadow: 0 8px 32px rgba(0, 255, 198, 0.3); */
            backdrop-filter: blur(12px);
            border: 1px solid rgba(0, 255, 198, 0.4);
            width: 100%;
            margin: auto;
        }
    </style>
    <div class='glass-title'>Vegetation Index Analysis Dashboard</div>
    <br><br>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("""
    <div style="height: 1px; background: linear-gradient(to right, #ff7e5f, #feb47b); margin-bottom: 10px;"></div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');

        [data-testid="stSidebar"] {
            background: rgba(119,158,203,0.1) ;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            /* box-shadow: 0 4px 30px rgba(212, 231, 245, 0.5);*/
            border-right: 1px solid rgba(255, 255, 255, 0.3);
        }

        .sidebar-title {
            text-align: center;
            font-size: 20px;
            font-family: 'Orbitron', sans-serif;
            color: #D4E7F5; 
            padding: 10px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.1); /* Subtle glass effect */
            /* box-shadow: 0 8px 32px rgba(255, 255, 255, 0.2);*/
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            margin-bottom: 19px;
        }
        
    </style>
    <div class='sidebar-title'>Vegetation Index Parameters</div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("""
    <div style="height: 1px; background: linear-gradient(to right, #ff7e5f, #feb47b); margin-bottom: 10px;"></div><br>
""", unsafe_allow_html=True)


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
    conn = sqlite3.connect("NDVI_Database.db")
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
def save_histogram_to_db(report_name, location, data):
    conn = sqlite3.connect("NDVI_Database.db")
    cursor = conn.cursor()

    # Convert DataFrame to CSV string as sqlite dont support dataframe
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()

    # Insert into the histogram_reports table
    cursor.execute("INSERT INTO histogram_reports (report_name, location, data) VALUES (?, ?, ?)", 
                   (report_name, location, csv_content))
    
    conn.commit()
    conn.close()
    st.success(f"Histogram Report '{report_name}' has been saved to the database.")

# Function to save heatmap data to the database
def save_heatmap_to_db(report_name, location, data,report_type):
    conn = sqlite3.connect("NDVI_Database.db")
    cursor = conn.cursor()
    
     # Insert each row into the table
    for _, row in data.iterrows():
        cursor.execute("""
            INSERT INTO Heatmap_Reports(report_name, location, latitude, longitude, ndvi_value, report_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (report_name, location, row['Latitude'], row['Longitude'], row[f'{index} Value'], report_type))

    conn.commit()
    conn.close()
    st.success(f"Heatmap Report '{report_name}' has been saved to the database.")

# Function to save heatmap data to the database
def save_surfaceData_to_db(report_name, location, data,report_type):
    conn = sqlite3.connect("NDVI_Database.db")
    cursor = conn.cursor()
    
     # Insert each row into the table
    for _, row in data.iterrows():
        cursor.execute("""
            INSERT INTO Surface_data_report(report_name, location, latitude, longitude, ndvi_value, report_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (report_name, location, row['Latitude'], row['Longitude'], row[f'{index} Value'], report_type))

    conn.commit()
    conn.close()
    st.success(f"SurfaceData Report '{report_name}' has been saved to the database.")

# feedback 
def save_feedback_to_db(feedback_text):
    conn = sqlite3.connect("NDVI_Database.db")
    cursor = conn.cursor()
    
    # Insert feedback
    cursor.execute("INSERT INTO User_Feedback (feedback) VALUES (?)", (feedback_text,))
    conn.commit()
    conn.close()

    print("Your feedback has been saved! Thankyou for your efforts !! 🎉")

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
        st.rerun()  

st.subheader("📍 INTERACTIVE LOCATION SELECTION")


# Interactive map for user
folium_map = folium.Map(location=selected_coordinates, zoom_start=6)
marker = folium.Marker(selected_coordinates, popup="Selected Location", draggable=True)
marker.add_to(folium_map)
folium_map.add_child(MeasureControl())
folium_map.add_child(MousePosition())

map_data = st_folium(folium_map, width=1100, height=600, key="location_map")

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
    size=(512, 512),
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
            st.markdown("<h3 style='text-align: left; color: #b2beb5;'>True Colour Image</h3>", unsafe_allow_html=True)
            # Image and download button placed tightly
            st.image(true_color, use_container_width=True)
            
            # informatin text box
            st.markdown(f"<p style='font-size:21px;'><b>{image_type}: {image_descriptions[image_type]['short']}</b></p>", unsafe_allow_html=True)

            # Download button - True Color Image
            st.download_button(
                label="Download True Color Image",
                data =save_image_as_png(true_color),
                file_name="true_color_image.png",
                mime="image/png"
            )

        else:
            st.markdown("<h3 style='text-align: left; color: #b2beb5;'>Infrared Colour Image</h3>", unsafe_allow_html=True)
            st.image(infrared_image, use_container_width=True)
            
            # information text box
            st.markdown(f"<p style='font-size:21px;'><b>{image_type}: {image_descriptions[image_type]['short']}</b></p>", unsafe_allow_html=True)

            # Download button - infra Color Image
            st.download_button(
                label="Download infrared Color Image",
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
        st.markdown(f"<h3 style='text-align: left; color: #b2beb5;'>{index} Image </h3>", unsafe_allow_html=True)
        st.image(vegetation_index_color, use_container_width=True)
        # info textbox
        st.markdown(f"<p style='font-size:21px;'><b>{index}: {image_descriptions[index]['short']}</b></p>", unsafe_allow_html=True)

        # Add download button directly below NDVI/EVI/SAVI Image
        st.download_button(
            label=f"Download {index} Image",
            data=vegetation_bytes,
            file_name=f"{index.lower()}_image.png",
            mime="image/png"
        )

        st.markdown("<hr style='border: 1px solid #f0ffff; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)


        # --------- Time Line Graph ---------

        # Generate Date Range
        dates = [start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        actual_ndvi_values = calculate_ndvi(nir_band, red_band).flatten()[:len(dates)]
        
        st.markdown(f"<h3 style='text-align: left; color: #b2beb5;'>{index} Time-Line Graph</h3>", unsafe_allow_html=True)
        
       
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates, y=actual_ndvi_values, 
            mode='lines+markers', 
            line=dict(color='green', width=2),
            marker=dict(size=8, color='red'),
            name=f"{index} Values"
        ))

        fig.update_layout(
            title=dict(
                text=f"🌿 {index} Trend for {st.session_state.location_input}",
                font=dict(size=18, color="#4CAF50"),
                x=0.5,  # Centers the title
                xanchor='center'  # Ensures alignment
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
            legend=dict(font=dict(size=13)),
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent Background
        )

        st.plotly_chart(fig, use_container_width=True)

        # Generate CSV 
        graph_csv = pd.DataFrame({'Date': dates, f'{index}': actual_ndvi_values}).to_csv(index=False)
        # Add CSV download button 
        st.download_button(
            label=f"Download {index} Graph Data as CSV",
            data=graph_csv,
            file_name=f"{index.lower()}_graph_data.csv",
            mime="text/csv"
        )

        # Graph DB Saving Button
        if st.button(f"Save {index} Graph Data to Database"):
            save_graph_to_db(
                f"{index}_Graph_{st.session_state.location_input}", 
                st.session_state.location_input, 
                pd.DataFrame({'Date': dates, 'NDVI Value': actual_ndvi_values})
            )

        st.markdown("<hr style='border: 1px solid #f0ffff; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)


        # --------- HISTOGRAM ---------

        st.markdown(f"<h3 style='text-align: left; color: #b2beb5;'>{index} Value Distribution</h3>", unsafe_allow_html=True)

        # Frequency distribution histogram
        freq_index = vegetation_index.flatten()
        freq_index = freq_index[~np.isnan(freq_index)]  # Remove NaN values

        # Convert data to a Pandas DataFrame
        dynamic_col_name = f"{index} Value"
        df = pd.DataFrame({dynamic_col_name : freq_index}) 

        # Create an interactive histogram using Plotly
        fig = px.histogram(df, x=dynamic_col_name, nbins=50, color_discrete_sequence=['green'])
        fig.update_layout(
            title=dict(
                text=f"🌿 {index} Frequency Distribution for {st.session_state.location_input} ",
                font=dict(size=20, color="#4CAF50"),
                x=0.5,  
                xanchor='center'  
            ),
            xaxis_title="NDVI Value",
            yaxis_title="Frequency",
            bargap=0.1
        )

        # Display Plotly chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Generate CSV
        histogram_csv = df.to_csv(index=False)

        # Add CSV download button 
        st.download_button(
            label=f"Download {index} Histogram Data as CSV",
            data=histogram_csv,
            file_name=f"{index} histogram_data.csv",
            mime="text/csv"
        )

        # Histogram DB Saving Button
        if st.button(f"Save {index} Histogram Data to Database"):
            save_histogram_to_db(f"{index}_Histogram_{st.session_state.location_input}", 
                                st.session_state.location_input, 
                                pd.DataFrame({'Value': freq_index}))
            
        st.markdown("<hr style='border: 1px solid #f0ffff; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)

        # -------- HEATMAP --------

        st.markdown(f"<h3 style='text-align: left; color: #b2beb5;'>{index} Heatmap Visualization</h3>", unsafe_allow_html=True)

        # NDVI Data 
        vegetation_index = np.nan_to_num(vegetation_index, nan=0)  # Replace NaN with 0 or -1 as dataloss may happen
        vegetation_index = np.flipud(vegetation_index)  # Flip top-to-bottom
        # ndvi_normalized = (vegetation_index + 1) / 2  # Shifts -1 to 1 → 0 to 1
    
        # Create a Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=vegetation_index,
            # z=ndvi_normalized,
            colorscale=[
                    [0.0, 'lightblue'],     # -1 NDVI (Water)
                    [0.3, 'yellow'],        #  0 NDVI (Bare Land)
                    [0.5, 'orange'],       #  0.3 NDVI (Sparse Vegetation)
                    [0.8, 'green'],        #  0.6 NDVI (Moderate Vegetation)
                    [1.0, 'darkgreen']      #  1 NDVI (Dense Vegetation)
            ],            
            colorbar=dict(title="NDVI Values Range"),
            zmin=-1, zmax=1,
            # zmin=0, zmax=1, #normalized range
        ))

        fig.update_layout(
            title=dict(
                text=f"🌿 {index} Heatmap for {st.session_state.location_input} ",
                font=dict(size=20, color="#4CAF50"),
                x=0.5,  # Centers the title
                xanchor='center'  # Ensures alignment
            ),
            xaxis_title=f"Longitude : {longitude}",
            yaxis_title=f"Latitude : {latitude}",
            height=700, 
            width=1000
        )
        st.plotly_chart(fig, use_container_width=True)

        # Get Grid Dimensions
        height, width = vegetation_index.shape

        # Define latitude and longitude range
        lat_range = 0.05  
        lon_range = 0.05 

        # Set min and max values for latitude and longitude
        latitude_min, latitude_max = latitude - lat_range, latitude + lat_range
        longitude_min, longitude_max = longitude - lon_range, longitude + lon_range

        # Create Longitude and Latitude Arrays
        lon_values = np.linspace(longitude_min, longitude_max, width)
        lat_values = np.linspace(latitude_min, latitude_max, height)
        lat_values_flip = np.flip(lat_values) 

        # Create Meshgrid for Proper Mapping
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values_flip)

        # Flatten the Data
        flattened_ndvi = vegetation_index.flatten()
        flattened_lat = lat_grid.flatten()
        flattened_lon = lon_grid.flatten()

        # Convert to Pandas DataFrame
        df = pd.DataFrame({
            'Latitude': flattened_lat,
            'Longitude': flattened_lon,
            f'{index} Value': flattened_ndvi
        })
        # Generate CSV File
        csv_data = df.to_csv(index=False)

        # Add CSV Download Button
        st.download_button(
            label=f"Download {index} Histogram Data as CSV",
            data=csv_data,
            file_name=f"{index} heatmap_data.csv",
            mime="text/csv"
        )
        # Heatmap DB Saving Button
        if st.button(f"Save {index} Heatmap Data to Database"):
            save_heatmap_to_db(
                f"{index}_heatmap_{st.session_state.location_input}", 
                st.session_state.location_input, 
                data=df,
                report_type=index
            )

        st.markdown("<hr style='border: 1px solid #f0ffff; margin-top:20px; margin-bottom:20px;'>", unsafe_allow_html=True)

        # ------------ 3D Surface Plot -----------

        st.markdown(f"<h3 style='text-align: left; color: #b2beb5;'>{index} 3D Surface Plot</h3>", unsafe_allow_html=True)

        veg_data_surface = vegetation_index
        
        colorscale_reversed = "YlGnBu_r"

        # Create 3D Surface Plot
        fig = go.Figure(data=[go.Surface(
            x=lon_grid, y=lat_grid, z=veg_data_surface, 
            colorscale=colorscale_reversed, 
            cmin=-1, cmax=1
        )])

        fig.update_layout(
            title=dict(
                text=f"🌿 {index} 3D Surface Visualization for {st.session_state.location_input} ",
                font=dict(size=20, color="#4CAF50"),
                x=0.5,  
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title=f"Longitude : {longitude}",
                    gridcolor="lightgray",  
                    gridwidth=0.4  
                ),
                yaxis=dict(
                    title=f"Latitude : {latitude}",
                    gridcolor="lightgray",
                    gridwidth=0.4
                ),
                zaxis=dict(
                    title=f"{index} Value",
                    range=[-1, 1],
                    gridcolor="lightgray",
                    gridwidth=0.4
                )
            ),
            width=1000,  
            height=800  
        )

        st.plotly_chart(fig, use_container_width=True)

        # Generate CSV File
        csv_data = df.to_csv(index=False)

        # Add CSV Download Button
        st.download_button(
            label=f"Download {index} Surface Plot Data as CSV",
            data=csv_data,
            file_name=f"{index} 3D_Surface_Plot_data.csv",
            mime="text/csv"
        )
        # 3D Surface Plot DB Saving Button
        if st.button(f"Save {index} Surface Plot Data to Database"):
            save_surfaceData_to_db(
                f"{index}_3D_Surface_Plot_{st.session_state.location_input}", 
                st.session_state.location_input, 
                data=df,
                report_type=index
            )

st.sidebar.markdown("""
    <br><div style="height: 1px; background: linear-gradient(to right, #ff7e5f, #feb47b);"></div><br>
""", unsafe_allow_html=True)

# st.sidebar.markdown(
#     "<p style='text-align: center; font-size: 13px; color: white;'><b>NDVI | Sentinel Hub</b></p>",
#     unsafe_allow_html=True
# )

st.sidebar.markdown(
    """
    <style>
    .custom-link {
        text-align: center;
        font-size: 15px;
        color: white !important; 
        text-decoration: none !important; 
        transition: all 0.3s ease-in-out;
    }
    .custom-link:hover {
        font-size: 16px; /* Slightly bigger on hover */
        text-decoration: underline !important; /* Underline appears on hover */
        color: #FFFFFF;  /* Light green glow effect */
        text-shadow: 0px 0px 8px ##FFFFFF;
    }
    </style>
    
    <p style="text-align: center;">
        <a class="custom-link" href="https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index" target="_blank">
            NDVI
        </a> 
        &nbsp;&nbsp;&nbsp;  <!-- Extra spaces -->
        | 
        &nbsp;&nbsp;&nbsp;  <!-- Extra spaces -->
        <a class="custom-link" href="https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndvi" target="_blank">
            Sentinel Hub
        </a>
    </p>
    """,
    unsafe_allow_html=True
)


st.markdown("---")  
st.subheader("💡 Feedback")
feedback = st.text_area("Got ideas or feedback? We'd love to hear from you—together, we can make this project even better.")

if st.button("Submit Feedback"):
    if feedback:
        st.success("Thank you for your feedback! 🙌")
        save_feedback_to_db(feedback)
    else:
        st.warning("Please enter your feedback before submitting.")
