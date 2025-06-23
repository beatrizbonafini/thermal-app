import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import control
import tempfile
import flyr
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
import requests
import cv2
import base64
from PIL import Image
import io
from img_processing import ProcessedImage
from backend import get_image_inference

url = "https://41f3-34-124-238-99.ngrok-free.app/segmentar/" 

print(sys.executable)

st.set_page_config(page_title="Thermography Image Mice Analyser", page_icon="🐭", layout="wide")

st.sidebar.title("List")

st.title("Thermography Image Mice Analyser")

tab1, tab2 = st.tabs(["Single Image Processing", "Patch-Based Processing"])

with tab1:

    uploaded_file = st.file_uploader("Upload a image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        
        # Salvar o arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        begin = time.time()

        control.get_thermal_gray(temp_file_path)
        
        with st.spinner("Sending image to server and waiting for response..."):

            with open("gray_image.jpg", 'rb') as f:
                files = {'file': f}
                response = requests.post(url, files=files)
        
            if response.status_code == 200:
                data = response.json()
                img_bytes = base64.b64decode(data["image_b64"])
                img = Image.open(io.BytesIO(img_bytes))
            else:
                st.error(f"Erro: {response.status_code} - {response.text}")

            end = time.time()
            st.write(f"Tempo de processamento: {end - begin:.2f} segundos")
        
        # Apresentando metadados da imagem
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        exif_data = image._getexif()
        if exif_data is not None:
            for tag_id, valor in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == 'MakerNote' or tag == 'ComponentsConfiguration':
                    continue
                #st.write(f"{tag}: {valor}")
        else:
            st.write("No EXIF data found.")

        #st.image(image, caption="Uploaded Image", use_column_width=True)
        #st.write("Image shape:", image_array.shape)

        st.caption(f"General Statistics")
        thermal_image = control.get_thermal_matrix(temp_file_path)
        rgb_image = control.get_rgb(temp_file_path)
        
        dataframe = pd.DataFrame(control.thermal_stats(np.array(thermal_image), exif_data))
        st.dataframe(dataframe, use_container_width=True)

        # PRECISO AJUSTAR ESSA PARTE PARA MOSTRAR AS IMAGENS E GRÁFICOS

        col1, col2, col3, col4 = st.columns(4)

        images_dict = {
            'original': {
                'value': image,
                'title': 'Original Image',
            },
            'rgb': {
                'value': rgb_image,
                'title': 'RGB Image',
            },
            'thermal': {
                'value': image_array,
                'title': 'Thermal Image',
            },
            'histogram': {
                'value': control.image_histogram(thermal_image),
                'title': 'Thermal Histogram'
            },
            'head_image_mask': {
                'value': control.crop_mask_from_image(thermal_image, data['masks'][0]),
                'title': 'Head Image Mask',
            },
            'head_image_bbox': {
                'value': control.crop_boxes_from_image(thermal_image, data['boxes'][0]),
                'title': 'Head Image BBox',
            },
            'head_image_mask_histogram': {
                'value': None,
                'title': 'Head Image BBox',
            },
            'implant_image_mask': {
                'value': control.crop_mask_from_image(thermal_image, data['masks'][1]),
                'title': 'Head Image Mask',
            },
            'implant_image_bbox': {
                'value': control.crop_mask_from_image(thermal_image, data['masks'][1]),
                'title': 'Head Image BBox',
            },
            'implant_image_mask_histogram': {
                'value': None,
                'title': 'Head Image BBox',
            },
        }

        col1, col2, col3, col4 = st.columns(4)
        
        col1.caption(f"RGB Image {rgb_image.shape}")
        col1.image(rgb_image, use_column_width=True)
        
        col2.caption(f"Original Image {image_array.shape}")
        col2.image(image, use_column_width=True)
        
        col3.caption(f"Thermal Image {thermal_image.shape}")
        fig, ax = plt.subplots(dpi=100)
        cax = ax.imshow(thermal_image, cmap='inferno')
        fig.colorbar(cax)
        col3.image(control.matplotlib_figure_to_image(fig), use_column_width=True)

        col4.caption("Thermal Histogram")
        fig_histogram = control.image_histogram(thermal_image)
        col4.image(control.matplotlib_figure_to_image(fig_histogram), use_column_width=True)

        head_image_mask = control.crop_mask_from_image(thermal_image, data['masks'][0])
        head_image = control.crop_boxes_from_image(head_image_mask, data['boxes'][0])
        col1.caption(f"Head Image {head_image.shape}")
        fig, ax = plt.subplots()
        cax = ax.imshow(head_image, cmap='inferno')
        fig.colorbar(cax)
        col1.image(control.matplotlib_figure_to_image(fig), use_column_width=True)

        col2.caption("Thermal Histogram")
        fig_histogram = control.image_histogram(head_image_mask)
        col2.image(control.matplotlib_figure_to_image(fig_histogram), use_column_width=True)

        implant_image_mask = control.crop_mask_from_image(thermal_image, data['masks'][1])
        implant_image = control.crop_boxes_from_image(implant_image_mask, data['boxes'][1])
        col3.caption(f"Mask Implant Image {implant_image.shape}")
        fig, ax = plt.subplots()
        cax = ax.imshow(implant_image, cmap='inferno')
        fig.colorbar(cax)
        col3.image(control.matplotlib_figure_to_image(fig), use_column_width=True)

        col4.caption("Thermal Histogram")
        fig_histogram = control.image_histogram(implant_image_mask)
        col4.image(control.matplotlib_figure_to_image(fig_histogram), use_column_width=True)

        calculate_donut_roi = control.calculate_donut_roi(images_dict['implant_image_mask']['value'], thermal_image)
        donut_image = control.crop_boxes_from_image(calculate_donut_roi, data['boxes'][0])

        col1.caption(f"Calculate Donut ROI {calculate_donut_roi.shape}")
        fig, ax = plt.subplots(dpi=100)
        cax = ax.imshow(donut_image, cmap='inferno')
        fig.colorbar(cax)
        col1.image(control.matplotlib_figure_to_image(fig), use_column_width=True)

        donut_histogram = control.image_histogram(donut_image)
        col2.caption("Donut Histogram")
        fig_histogram = control.image_histogram(donut_image)
        col2.image(control.matplotlib_figure_to_image(fig_histogram), use_column_width=True)

        ## LINHA 4

with tab2:

    uploaded_files = st.file_uploader(
        "Upload all files from a folder (select multiple)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    df_info = []

    for file in uploaded_files:

        st.write(f'Processing {file.name}')

        file_bytes = file.read()
    
        unpacked_file = control.unpack_from_bytes(file_bytes)
        thermal_matrix = unpacked_file[0]
        optical_image = unpacked_file[1]
        grayscale_image = unpacked_file[2]

        original_image = Image.open(file)
        exif_data = original_image._getexif()
    
        buffer = io.BytesIO()
        grayscale_image.save(buffer, format="PNG")
        buffer.seek(0)

        with st.spinner(f'Sending {file.name} to server ...'): data = get_image_inference({'file': buffer})

        #img_bytes = base64.b64decode(data["image_b64"])
        #image = Image.open(io.BytesIO(img_bytes))
        
        img_obj = ProcessedImage(original_image=original_image,
                                 optical_image=optical_image,
                                 thermal_matrix=thermal_matrix,
                                 grayscale=grayscale_image,
                                 animal_id="Sheep42", 
                                 metadata=exif_data,
                                 masks=data['masks'],
                                 boxes=data['boxes'],
                                 classes=data['classes'])

        st.write(f"Timestamp: {img_obj.get_metadata()['DateTime']}")
        cols = st.columns(5)
        cols[0].image(img_obj.optical_image, caption="Optical Image")
        cols[1].image(img_obj.original_image, caption="Original Image (MSX)")   
        cols[2].image(img_obj.grayscale, caption="Grayscale Image")
        cols[3].image(img_obj.get_image_bboxes(class_id=0), caption="BBox Head")
        cols[4].image(img_obj.get_image_bboxes(class_id=1), caption="BBox Implant")

        df_info.append(img_obj.get_thermal_stats())

    if len(df_info) != 0:
        st.markdown('---')

        st.subheader("General Statistics")

        df_total = pd.concat(df_info, ignore_index=True)
        st.dataframe(df_total, use_container_width=True)
    
        st.markdown("#### Head")
        st.write('EM CONSTRUÇÃO')

        st.markdown("#### Implant")
        st.write('EM CONSTRUÇÃO')



