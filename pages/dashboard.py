import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import plotly.express as px
import plotly.graph_objects as go
import uuid
from datetime import datetime
from PIL import Image
import io
import cv2

from backend.instace_segmentation import InstanceSegmentationPredictor
import control

# --- Page Configuration ---
st.set_page_config(
    page_title="Thermography Analysis System",
    #page_icon="🌡️",
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
                           'image_bytes': None, # Example data has no image bytes
                           'regions': [
                                { 'id': 'r1', 'name': 'Upper Back R', 'color': 'red', 'points': np.array([[0.55, 0.2], [0.85, 0.2], [0.8, 0.45], [0.55, 0.4]]), 't_avg': 34.8, 't_max': 35.2, 'deltaT': 0.9, 'histogram': [2, 5, 10, 4, 1] },
                                { 'id': 'r2', 'name': 'Upper Back L', 'color': 'orange', 'points': np.array([[0.45, 0.2], [0.15, 0.2], [0.2, 0.45], [0.45, 0.4]]), 't_avg': 33.9, 't_max': 34.3, 'deltaT': -0.9, 'histogram': [4, 8, 5, 2, 0] },
                                { 'id': 'r3', 'name': 'Lower Back R', 'color': 'green', 'points': np.array([[0.55, 0.5], [0.8, 0.55], [0.75, 0.75], [0.55, 0.7]]), 't_avg': 35.1, 't_max': 35.5, 'deltaT': 0.2, 'histogram': [1, 3, 7, 9, 3] },
                                { 'id': 'r4', 'name': 'Lower Back L', 'color': 'blue', 'points': np.array([[0.45, 0.5], [0.2, 0.55], [0.25, 0.75], [0.45, 0.7]]), 't_avg': 34.9, 't_max': 35.2, 'deltaT': -0.2, 'histogram': [2, 4, 8, 6, 2] },
                           ]
                        }
                    }
                }
            }
        }
    }

# --- Functions ---


def read_and_unpack_image(uploaded_image) -> dict:
    
    try:
        file_bytes = uploaded_image.read()
        unpacked_file = control.unpack_from_bytes(file_bytes)
        
        thermal_matrix = unpacked_file[0]
        optical_image = unpacked_file[1]
        grayscale_image = unpacked_file[2]

        original_image = Image.open(uploaded_image)
        exif_data = original_image._getexif()
    
        buffer = io.BytesIO()
        grayscale_image.save(buffer, format="PNG")
        buffer.seek(0)

        return {
                'buffer': buffer,
                'thermal': thermal_matrix,
                'gray': grayscale_image,
                'optical': optical_image,
                'original': original_image, 
                'metadata': exif_data
            }
      
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return None

def mask_to_polygon(mask: np.ndarray) -> list:
    """
    Convert a binary mask to a polygon representation.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return []
    
    main_contour = max(contours, key=cv2.contourArea)
    polygon_points = np.squeeze(main_contour).tolist()

    if not isinstance(polygon_points[0], list):
        polygon_points = [polygon_points]

    return polygon_points

def segmentation_and_metrics(uploaded_image: bytes) -> list:
    """
    Segmentation process for a single image.
    This function returns a fixed set of regions.
    """
    server_config = InstanceSegmentationPredictor(model_weights_path="models/implant/model_final.pth", class_names=["head", "implant"])
    unpacked_image = read_and_unpack_image(uploaded_image)
    data_segmentation = server_config.predict(unpacked_image['gray'])
    masks = data_segmentation['masks']
    classes = data_segmentation['classes']
    
    w, h = unpacked_image['gray'].size
    polygon_boxes = []

    for mask, label in zip(masks, classes):
        polygon_points = mask_to_polygon(np.array(mask))
        if not polygon_points:
            continue
        polygon_boxes.append((polygon_points, label))
    
    all_polygons = []
    i = 0
    for polygon in polygon_boxes:
        points = polygon[0]
        label = 'Head' if polygon[1] == 0 else 'Implant'
        color = 'green' if label == 'Head' else 'blue'
        histogram = control.get_histogram(unpacked_image['thermal'])
        data = { 
            'id': 'r1', 
            'name': label, 
            'color': color, 
            'points': points,
            't_avg': 34.8, 
            't_max': 35.2, 
            'deltaT': 0.9, 
            'histogram': histogram
        }
    
        all_polygons.append(data)
        i = i + 1

    template_regions = st.session_state.patient_data['p001']['studies']['study_20250620']['images']['img01']['regions']
    return all_polygons

def create_thermal_image_plotly(palette, opacity, show_regions, image_data, selected_region_id):
    """
    Gera a imagem térmica com sobreposição de regiões usando Plotly,
    permitindo hover para ver os valores dos pixels.
    """
    img_array = None
    fig = None

    # 1. Tentar carregar a imagem real. Se não houver, usar um gradiente.
    if image_data.get('image_bytes'):
        try:

            ####### ESTOU TENTANDO RESOLVER ISSO AQUI ########
            
            #aux = read_and_unpack_image(io.BytesIO(image_data['image_bytes']))
            #aux_img = Image.open(aux['gray'])
            #print(type(aux_img))
            
            #######
            
            img = Image.open(io.BytesIO(image_data['image_bytes']))
            img_array = np.array(img)
        except Exception as e:
            st.error(f"Não foi possível ler o arquivo de imagem. Erro: {e}")
            # Retorna uma figura vazia em caso de erro
            return go.Figure()
    else:
        # Fallback para dados de exemplo se não houver imagem
        img_array = np.linspace(0, 255, 300*300, dtype=np.uint8).reshape(300, 300)

    # 2. Criar a figura base com px.imshow.
    # Esta é a parte mágica que habilita o hover nos pixels.
    # O hovertext mostrará x, y e o valor da cor (intensidade).
    fig = px.imshow(img_array, color_continuous_scale=palette)

    # 3. Desenhar as regiões, se habilitado
    if show_regions and image_data.get('regions'):
        for region in image_data['regions']:
            points = region['points']
            # O Plotly espera um caminho SVG para desenhar a forma
            path = 'M ' + ' L '.join([f'{p[0]},{p[1]}' for p in points]) + ' Z'

            # Adiciona o polígono da região
            fig.add_shape(
                type="path",
                path=path,
                fillcolor=region['color'],
                opacity=opacity,
                line=dict(color='white', width=1.5)
            )

            # Se a região for a selecionada, adiciona o destaque
            if region['id'] == selected_region_id:
                fig.add_shape(
                    type="path",
                    path=path,
                    fillcolor='rgba(0,0,0,0)',  # Preenchimento transparente
                    line=dict(color='yellow', width=3)
                )

    # 4. Limpar o layout para parecer uma imagem pura
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),      # Remove margens
        xaxis_visible=False, yaxis_visible=False, # Oculta os eixos
        coloraxis_showscale=False              # Oculta a barra de cores
    )

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
new_patient_name = st.sidebar.text_input("Animal Name for the Study")
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
            segmented_regions = segmentation_and_metrics(uploaded_file)
            new_study['images'][image_id] = {
                'fileName': uploaded_file.name,
                'image_bytes': uploaded_file.getvalue(), # Store the actual image bytes
                'regions': segmented_regions
            }
        
        st.session_state.patient_data[patient_id]['studies'][study_id] = new_study
        st.sidebar.success(f"Study with {len(uploaded_files)} image(s) processed for '{new_patient_name}'!")
    else:
        st.sidebar.error("Please provide a patient name and at least one image.")

st.sidebar.divider()
st.sidebar.subheader("Analyze Existing Study")

# Step 1: Select Animal
patient_names = {data['name']: patient_id for patient_id, data in st.session_state.patient_data.items()}
selected_patient_name = st.sidebar.selectbox(
    "1. Select Animal",
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
        #image_data_aux = read_and_unpack_image(image_data)
        #print(type(image_data))

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
        
        fig = create_thermal_image_plotly(palette, opacity, show_regions, image_data, selected_region_id)
        st.plotly_chart(fig, use_container_width=True)


    with col2:
        st.subheader("Metrics and Regions")
        st.markdown(f"**Animal:** {patient['name']} | **Study:** {selected_study_id}")
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
            hist_df = pd.DataFrame({'Temperature Range (°C)': region_data['histogram'][1], 'Pixel Count': region_data['histogram'][0]}).set_index('Temperature Range (°C)')
            st.bar_chart(hist_df)
            st.caption(f"Showing distribution for: **{selected_region_name}**")
        else:
            st.info("Highlight a region to see its temperature distribution.")
else:
    st.info("⬅️ To begin, load a new study or select an existing patient and study from the sidebar.")