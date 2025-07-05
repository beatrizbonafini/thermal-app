import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import uuid
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Thermography Analysis System",
    page_icon="🌡️",
    layout="wide"
)

# --- Initial Data and Session Management ---
# The structure now supports multiple studies per patient.
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {
        'p001': {
            'name': 'Ana Silva (Example)',
            'studies': {
                'study_20250620': {
                    'studyDate': '2025-06-20',
                    'images': {
                        'img01': {
                           'fileName': 'Full_Back.png',
                           'regions': [
                                { 'id': 'r1', 'name': 'Dorso Superior D', 'color': 'red', 'points': np.array([[0.55, 0.2], [0.85, 0.2], [0.8, 0.45], [0.55, 0.4]]), 't_avg': 34.8, 't_max': 35.2, 'deltaT': 0.9, 'histogram': [2, 5, 10, 4, 1] },
                                { 'id': 'r2', 'name': 'Dorso Superior E', 'color': 'orange', 'points': np.array([[0.45, 0.2], [0.15, 0.2], [0.2, 0.45], [0.45, 0.4]]), 't_avg': 33.9, 't_max': 34.3, 'deltaT': -0.9, 'histogram': [4, 8, 5, 2, 0] },
                                { 'id': 'r3', 'name': 'Lombar D', 'color': 'green', 'points': np.array([[0.55, 0.5], [0.8, 0.55], [0.75, 0.75], [0.55, 0.7]]), 't_avg': 35.1, 't_max': 35.5, 'deltaT': 0.2, 'histogram': [1, 3, 7, 9, 3] },
                                { 'id': 'r4', 'name': 'Lombar E', 'color': 'blue', 'points': np.array([[0.45, 0.5], [0.2, 0.55], [0.25, 0.75], [0.45, 0.7]]), 't_avg': 34.9, 't_max': 35.2, 'deltaT': -0.2, 'histogram': [2, 4, 8, 6, 2] },
                           ]
                        }
                    }
                }
            }
        }
    }

# --- Functions ---

def simulate_segmentation_and_metrics(uploaded_image):
    """Simulates the segmentation process for a single image."""
    # Returns a fixed set of regions for the demo.
    # In the real world, you would call your AI model here.
    template_regions = st.session_state.patient_data['p001']['studies']['study_20250620']['images']['img01']['regions']
    return template_regions

def create_thermal_image(palette, opacity, show_regions, image_data, selected_region_id):
    """Generates the thermal image with a region overlay."""
    fig, ax = plt.subplots(figsize=(8, 8))
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect='auto', cmap=palette, extent=[0, 1, 0, 1], origin='lower')

    if show_regions and image_data:
        for region in image_data['regions']:
            polygon = Polygon(region['points'], closed=True, facecolor=region['color'], alpha=opacity, edgecolor='white', linewidth=1.5)
            ax.add_patch(polygon)
            if region['id'] == selected_region_id:
                highlight_polygon = Polygon(region['points'], closed=True, facecolor='none', edgecolor='yellow', linewidth=3)
                ax.add_patch(highlight_polygon)
    ax.set_axis_off()
    return fig

def format_delta_t(val):
    """Formats the Delta T value with colors for emphasis."""
    if val > 0.5:
        return "color: red; font-weight: bold;"
    elif val < -0.5:
        return "color: blue; font-weight: bold;"
    else:
        return "color: black;"

# --- UI ---

st.title("🌡️ Interactive Thermography Analysis System")

# --- Sidebar (Controls) ---
st.sidebar.header("Analysis Controls")

# Section to load a new study
st.sidebar.divider()
st.sidebar.subheader("Load New Study")
new_patient_name = st.sidebar.text_input("Patient Name for the Study")
uploaded_files = st.sidebar.file_uploader(
    "Select one or more image files", 
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

if st.sidebar.button("Process Study"):
    if uploaded_files and new_patient_name:
        # Find or create patient
        patient_id = None
        for pid, data in st.session_state.patient_data.items():
            if data['name'] == new_patient_name:
                patient_id = pid
                break
        
        if not patient_id:
            patient_id = f"user_{uuid.uuid4().hex[:6]}"
            st.session_state.patient_data[patient_id] = {'name': new_patient_name, 'studies': {}}
            
        # Create a new study with an ID based on date/time
        study_id = f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_study = {'studyDate': datetime.now().strftime('%Y-%m-%d'), 'images': {}}
        
        # Process each uploaded image
        for i, uploaded_file in enumerate(uploaded_files):
            image_id = f"img_{i+1:02d}"
            segmented_regions = simulate_segmentation_and_metrics(uploaded_file)
            new_study['images'][image_id] = {
                'fileName': uploaded_file.name,
                'regions': segmented_regions
            }
        
        st.session_state.patient_data[patient_id]['studies'][study_id] = new_study
        st.sidebar.success(f"Study with {len(uploaded_files)} image(s) processed for '{new_patient_name}'!")
    else:
        st.sidebar.error("Please provide a patient name and at least one image.")

st.sidebar.divider()
st.sidebar.subheader("Analyze Existing Study")

# Step 1: Select Patient
patient_names = {data['name']: patient_id for patient_id, data in st.session_state.patient_data.items()}
selected_patient_name = st.sidebar.selectbox(
    "1. Select Patient",
    options=list(patient_names.keys()),
    index=None,
    placeholder="Choose a patient..."
)

selected_study_id = None
if selected_patient_name:
    patient_id = patient_names[selected_patient_name]
    patient = st.session_state.patient_data[patient_id]
    
    # Step 2: Select Study
    study_options = {f"Study from {data['studyDate']}": study_id for study_id, data in patient['studies'].items()}
    if study_options:
        selected_study_key = st.sidebar.selectbox(
            "2. Select Study",
            options=list(study_options.keys())
        )
        selected_study_id = study_options[selected_study_key]

# If a study is selected, show the analysis interface
if selected_study_id:
    study_data = patient['studies'][selected_study_id]

    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.subheader("Image Viewer")
        # Step 3: Select Image within the Study
        image_options = {f"{img_id}: {data['fileName']}": img_id for img_id, data in study_data['images'].items()}
        selected_image_key = st.selectbox(
            "3. Select Image for Analysis",
            options=list(image_options.keys())
        )
        selected_image_id = image_options[selected_image_key]
        image_data = study_data['images'][selected_image_id]

        # Visualization controls
        st.sidebar.subheader("Visualization Tools")
        
        opacity = st.sidebar.slider("Region Opacity", 0.0, 1.0, 0.5, 0.05, key=f"opacity_{patient_id}_{selected_study_id}")
        palette = st.sidebar.selectbox("Color Palette", ["plasma", "jet", "inferno", "gray"], index=0, key=f"palette_{patient_id}_{selected_study_id}")
        
        show_regions = st.sidebar.checkbox("Show Regions", value=True, key=f"show_{patient_id}_{selected_study_id}")
        
        # Region selection for highlight
        region_names = {region['name']: region['id'] for region in image_data['regions']}
        region_names["None"] = None
        selected_region_name = st.selectbox(
            "Highlight a region:",
            options=list(region_names.keys()),
            index=len(region_names) - 1,
            key=f"select_region_{patient_id}_{selected_study_id}_{selected_image_id}"
        )
        
        selected_region_id = region_names[selected_region_name]
        
        fig = create_thermal_image(image, palette, opacity, show_regions, image_data, selected_region_id)
        st.pyplot(fig)

    with col2:
        st.subheader("Metrics and Regions")
        st.markdown(f"**Patient:** {patient['name']} | **Study:** {selected_study_id}")
        st.markdown(f"**Image:** {image_data['fileName']}")
        
        metrics_df = pd.DataFrame(image_data['regions'])[['name', 't_avg', 't_max', 'deltaT']]
        metrics_df.rename(columns={'name': 'Region', 't_avg': 'Avg T (°C)', 't_max': 'Max T (°C)', 'deltaT': 'ΔT (°C)'}, inplace=True)
        st.dataframe(
            metrics_df.style.map(format_delta_t, subset=['ΔT (°C)']).format("{:.1f}", subset=['Avg T (°C)', 'Max T (°C)', 'ΔT (°C)']),
            hide_index=True,
            use_container_width=True
        )
        
        st.subheader("Temperature Distribution")
        if selected_region_id:
            region_data = next(r for r in image_data['regions'] if r['id'] == selected_region_id)
            hist_df = pd.DataFrame({'Temperature Range (°C)': ['33-34', '34-35', '35-36', '36-37', '37-38'], 'Pixel Count': region_data['histogram']}).set_index('Temperature Range (°C)')
            st.bar_chart(hist_df)
            st.caption(f"Showing distribution for: **{selected_region_name}**")
        else:
            st.info("Highlight a region to see its temperature distribution.")
else:
    st.info("⬅️ To begin, load a new study or select an existing patient and study from the sidebar.")

