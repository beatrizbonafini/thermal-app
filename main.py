from typing import List, Union
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import uuid
from datetime import datetime
from PIL import Image
import io
import cv2

from backend.instace_segmentation import InstanceSegmentationPredictor
import control

st.set_page_config(
    page_title="Thermography Analysis System",
    #page_icon="🌡️",
    layout="wide"
)

if 'animal_data' not in st.session_state:
    st.session_state.animal_data = {
        # 'p001': {
        #     'name': 'Ana Silva (Example)',
        #     'studies': {
        #         'study_20250620': {
        #             'studyDate': '2025-06-20',
        #             'images': {
        #                 'img01': {
        #                    'fileName': 'Full_Back.png',
        #                    'image_bytes': None, # Example data has no image bytes
        #                    'regions': [
        #                         { 'id': 'r1', 'name': 'Upper Back R', 'color': 'red', 'points': np.array([[0.55, 0.2], [0.85, 0.2], [0.8, 0.45], [0.55, 0.4]]), 't_avg': 34.8, 't_max': 35.2, 'deltaT': 0.9, 'histogram': [2, 5, 10, 4, 1] },
        #                         { 'id': 'r2', 'name': 'Upper Back L', 'color': 'orange', 'points': np.array([[0.45, 0.2], [0.15, 0.2], [0.2, 0.45], [0.45, 0.4]]), 't_avg': 33.9, 't_max': 34.3, 'deltaT': -0.9, 'histogram': [4, 8, 5, 2, 0] },
        #                         { 'id': 'r3', 'name': 'Lower Back R', 'color': 'green', 'points': np.array([[0.55, 0.5], [0.8, 0.55], [0.75, 0.75], [0.55, 0.7]]), 't_avg': 35.1, 't_max': 35.5, 'deltaT': 0.2, 'histogram': [1, 3, 7, 9, 3] },
        #                         { 'id': 'r4', 'name': 'Lower Back L', 'color': 'blue', 'points': np.array([[0.45, 0.5], [0.2, 0.55], [0.25, 0.75], [0.45, 0.7]]), 't_avg': 34.9, 't_max': 35.2, 'deltaT': -0.2, 'histogram': [2, 4, 8, 6, 2] },
        #                    ]
        #                 }
        #             }
        #         }
        #     }
        # }
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
    
    polygons = []
    for mask, label in zip(masks, classes):
        polygon_points = mask_to_polygon(np.array(mask))
        if not polygon_points:
            continue
        polygons.append((polygon_points, np.array(mask) * unpacked_image['thermal'], label))
    
    all_polygons = []
    i = 0
    for points, roi, classe in polygons:
        thermal_stats = control.thermal_stats(roi[roi != 0])
        label = 'Head' if classe == 0 else 'Implant'
        color = 'green' if label == 'Head' else 'blue'
        histogram = control.get_histogram(roi)
        data = { 
            'id': f'r{i}', 
            'name': label, 
            'color': color, 
            'points': points,
            't_avg': thermal_stats['mean'], 
            't_max': thermal_stats['max'],
            't_min': thermal_stats['min'],
            't_std': thermal_stats['std'],
            't_median': thermal_stats['median'],
            't_var': thermal_stats['var'],
            'percentil_5': thermal_stats['percentil_5'],
            'percentil_95': thermal_stats['percentil_95'],
            'deltaT': 0.9, 
            'histogram': histogram
        }
        all_polygons.append(data)
        i = i + 1

    return all_polygons

def create_thermal_image_plotly(palette, opacity, show_regions, image_data, selected_region_id):
    img_array = None
    fig = None

    if image_data.get('image_bytes'):
        try:
            aux = read_and_unpack_image(io.BytesIO(image_data['image_bytes']))
            img_array = np.array(aux['thermal'])
        except Exception as e:
            st.error(f"Não foi possível ler o arquivo de imagem. Erro: {e}")
            return go.Figure()
    else:
        img_array = np.linspace(0, 255, 300*300, dtype=np.uint8).reshape(300, 300)

    fig = go.Figure(
       data = go.Heatmap(
            z=img_array,
            colorscale=palette,
            colorbar=dict(title=dict(text='Temperature (°C)', side='right')),
            showscale=True,
            hovertemplate=(
                "Coordinate x: %{x}<br>" +
                "Coordinate y: %{y}<br>" +
                "Temp: %{z:.2f} °C<extra></extra>"
            )
        ) 
    )
    fig.update_yaxes(autorange='reversed', scaleanchor="x", scaleratio=1)

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
                    line=dict(color='blue', width=1)
                )

    fig.update_layout(
        #margin=dict(l=0, r=0, t=0, b=0),      # Remove margens
        xaxis_visible=False, 
        yaxis_visible=False, # Oculta os eixos
        coloraxis_showscale=True,              # Oculta a barra de cores
        uirevision='persistent_vision' # Mantem o estado da UI
    )

    return fig

def create_histogram_plotly(hist_counts: Union[np.array, List],
                            bin_edges: Union[np.array, List],
                            percentiles_values: List[float] = [5, 95],
                            title: str = 'Histogram') -> go.Figure:
    
    df = pd.DataFrame({
        'Temperature Range (°C)': bin_edges,
        'Count': hist_counts
    })
    
    fig = px.bar(
        df,
        x='Temperature Range (°C)',
        y='Count',
        title=title,
        labels={'Temperature Range (°C)': 'Temperature (°C)', 'Pixel Count': 'Frequency'},
        color_discrete_sequence=["blue"]
    )
    fig.update_layout(showlegend=True)
    return fig

def plot_3d_thermal_chart(image_data, palette) -> go.Figure:
    """
    Generates an interactive 3D surface plot from a thermal data matrix.

    Args:
        thermal_image (np.ndarray): A 2D NumPy array with temperature values.

    Returns:
        go.Figure: A Plotly Figure object ready to be displayed.
    """

    if image_data.get('image_bytes'):
        try:
            aux = read_and_unpack_image(io.BytesIO(image_data['image_bytes']))
            img_array = np.array(aux['thermal'])
        except Exception as e:
            st.error(f"Não foi possível ler o arquivo de imagem. Erro: {e}")
            return go.Figure()

    if not isinstance(img_array, np.ndarray) or img_array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")


    height, width = img_array.shape
    x_coords = np.linspace(0, width - 1, width)
    y_coords = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = img_array

    fig = go.Figure(data=[
        go.Surface(
            x=X, 
            y=Y, 
            z=Z,
            colorscale=palette,  
            colorbar=dict(title=dict(text='Temperature (°C)', side='right')), 
            cmin=np.min(Z),
            cmax=np.max(Z),  
            hovertemplate=(
                "Coordinate X: %{x}<br>" +
                "Coordinate Y: %{y}<br>" +
                "Temperature: %{z:.2f}°C" + 
                "<extra></extra>"
            )
        )
    ])

    fig.update_layout(
        title='3D Visualization of Thermal Data',
        scene=dict(
            xaxis_title='X Coordinate (pixel)',
            yaxis_title='Y Coordinate (pixel)',
            zaxis_title='Temperature (°C)',
            # Optional: Adjust the axis aspect ratio for a better view
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode='manual'
        ),
        margin=dict(l=0, r=0, b=0, t=40) # Minimal margins
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

st.title("Interactive Thermography Analysis System")

# --- Sidebar (Controls) ---
st.sidebar.header("Analysis Controls")

# Section to load a new study
st.sidebar.divider()
st.sidebar.subheader("Load New Study")
new_animal_name = st.sidebar.text_input("Animal ID for the Study", help='Provide the identify number of animal in the study')
uploaded_files = st.sidebar.file_uploader(
    "Select one or more image files", 
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

if st.sidebar.button("Process Study"):
    if uploaded_files and new_animal_name:
        # Find or create animal
        animal_id = None
        for pid, data in st.session_state.animal_data.items():
            if data['name'] == new_animal_name:
                animal_id = pid
                break
        
        if not animal_id:
            animal_id = f"user_{uuid.uuid4().hex[:6]}"
            st.session_state.animal_data[animal_id] = {'name': new_animal_name, 'studies': {}}
            
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
        
        st.session_state.animal_data[animal_id]['studies'][study_id] = new_study
        st.sidebar.success(f"Study with {len(uploaded_files)} image(s) processed for '{new_animal_name}'!")
    else:
        st.sidebar.error("Please provide a animal id and at least one image.")

st.sidebar.divider()
st.sidebar.subheader("Analyze Existing Study")

# Step 1: Select Animal
animal_names = {data['name']: animal_id for animal_id, data in st.session_state.animal_data.items()}
selected_animal_name = st.sidebar.selectbox(
    "1. Select Animal",
    options=list(animal_names.keys()),
    index=None,
    placeholder="Choose a animal..."
)

selected_study_id = None
if selected_animal_name:
    animal_id = animal_names[selected_animal_name]
    animal = st.session_state.animal_data[animal_id]
    
    # Step 2: Select Study
    study_options = {f"Study from {data['studyDate']}": study_id for study_id, data in animal['studies'].items()}
    if study_options:
        selected_study_key = st.sidebar.selectbox(
            "2. Select Study",
            options=list(study_options.keys())
        )
        selected_study_id = study_options[selected_study_key]

# If a study is selected, show the analysis interface
if selected_study_id:
    study_data = animal['studies'][selected_study_id]

    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.subheader("Image Viewer")
        # Step 3: Select Image within the Study
        image_options = {f"{img_id}: {data['fileName']}": img_id for img_id, data in study_data['images'].items()}
        image_options['Temporal Analysis'] = None
        selected_image_key = st.selectbox(
            "3. Select Image for Analysis",
            options=list(image_options.keys())
        )

        # plotar grafico da progressao de temperatura
        if selected_image_key == 'Temporal Analysis':
            array_metric = []
            metrics = ['fileName', 'name', 't_avg', 't_max', 't_min', 't_std', 't_median', 't_var', 'percentil_5', 'percentil_95']
            for image_data in study_data['images'].items():
                for region in image_data[1]['regions']:
                    region['fileName'] = image_data[1]['fileName']
                    array_metric.append({key: region[key] for key in metrics})
            df = pd.DataFrame(array_metric)
            
            df = df.rename(columns={
                'fileName': 'Image',
                'name': 'Region',
                't_min': 'Minimum Temperature',
                't_avg': 'Average Temperature',
                't_max': 'Maximum Temperature'
            })
            
            all_regions = df['Region'].unique()
            selected_regions = st.sidebar.multiselect(
                "Select the Region(s)",
                options=all_regions,
                default=all_regions
            )

            metric_options = ['Minimum Temperature', 'Average Temperature', 'Maximum Temperature']
            selected_metric = st.sidebar.selectbox(
                "Select Temperature Metric",
                options=metric_options
            )

            if not selected_regions:
                st.warning("Please select at least one region in the sidebar to view the data.")
            else:
                # Filtra o DataFrame com base nas regiões selecionadas pelo usuário
                df_filtered = df[df['Region'].isin(selected_regions)]

                st.subheader(f"Evolution of {selected_metric}")

                # Cria o gráfico de linhas
                st.line_chart(
                    df_filtered,
                    x='Image',          # Eixo X: As imagens em sequência
                    y=selected_metric,   # Eixo Y: A métrica de temperatura escolhida
                    color='Region'       # Cor: Diferencia as linhas por região
                )

                # Mostra a tabela de dados brutos que foi usada para gerar o gráfico
                st.subheader("Detailed Data")
                st.dataframe(df_filtered)


        else:
            selected_image_id = image_options[selected_image_key]
            image_data = study_data['images'][selected_image_id]

            # Visualization controls
            st.sidebar.subheader("Visualization Tools")
            opacity = st.sidebar.slider("Region Opacity", 0.0, 1.0, 0.5, 0.05, key=f"opacity_{animal_id}_{selected_study_id}")
            palette = st.sidebar.selectbox("Color Palette", ["plasma", "jet", "inferno", "gray"], index=0, key=f"palette_{animal_id}_{selected_study_id}")
            show_regions = st.sidebar.checkbox("Show Regions", value=True, key=f"show_{animal_id}_{selected_study_id}")
            
            # Region selection for highlight
            region_names = {region['name']: region['id'] for region in image_data['regions']}
            region_names["None"] = None
            selected_region_name = st.selectbox(
                "Highlight a region:",
                options=list(region_names.keys()),
                index=len(region_names) - 1,
                key=f"select_region_{animal_id}_{selected_study_id}_{selected_image_id}"
            )
            selected_region_id = region_names[selected_region_name]
            
            fig = create_thermal_image_plotly(palette, opacity, show_regions, image_data, selected_region_id)
            st.plotly_chart(fig, use_container_width=True)

            fig_3d = plot_3d_thermal_chart(image_data, palette)
            st.plotly_chart(fig_3d, use_container_width=True)


    with col2:
        if selected_image_key != 'Temporal Analysis':
            st.subheader("Metrics and Regions")
            st.write(f"**Animal:** {animal['name']}")
            st.write(f"**Image:** {image_data['fileName']}")
            
            metrics_df = pd.DataFrame(image_data['regions'])[['name', 't_avg', 't_max', 't_min', 't_std', 't_median', 't_var', 'percentil_5', 'percentil_95']]
            metrics_df.rename(columns={
                'name': 'Region', 
                't_avg': 'Avg T (°C)', 
                't_max': 'Max T (°C)',
                't_min': 'Min T (°C)',
                't_std': 'Std Dev (°C)',
                't_median': 'Median T (°C)',
                't_var': 'Variance (°C²)',
                'percentil_5': '5th Percentile (°C)',
                'percentil_95': '95th Percentile (°C)'
                }, inplace=True)
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            st.subheader("Temperature Distribution")
            if selected_region_id:
                region_data = next(r for r in image_data['regions'] if r['id'] == selected_region_id)
                hist_df = pd.DataFrame({'Temperature Range (°C)': region_data['histogram'][1], 'Pixel Count': region_data['histogram'][0]}).set_index('Temperature Range (°C)')
                st.write(f"Showing distribution for: **{selected_region_name}**")
                st.plotly_chart(create_histogram_plotly(region_data['histogram'][0], region_data['histogram'][1]), use_container_width=True)
            else:
                st.info("Highlight a region to see its temperature distribution.")
else:
    st.info("⬅️ To begin, load a new study or select an existing animal and study from the sidebar.")