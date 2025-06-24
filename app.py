import streamlit as st
from PIL import Image
import numpy as np
import control
import matplotlib.pyplot as plt
import pandas as pd
import time
from PIL import Image
import io
import glob
import os
from img_processing import ProcessedImage, Base
from backend import InstanceSegmentationPredictor
from services.database.connection import init_db
import matplotlib.cm as cm
import matplotlib.colors as mcolors

session = init_db()
Base.metadata.create_all(session.bind)

st.set_page_config(page_title="Thermography Image Mice Analyser", 
                   page_icon="🐭", 
                   layout="wide")

st.title("Thermography Image Mice Analyser")

server_config = InstanceSegmentationPredictor(model_weights_path="models/implant/model_final.pth", 
                                              class_names=["head", "implant"])

tab1, tab2, tab3 = st.tabs(["Single Image Processing", 
                            "Patch-Based Processing", 
                            "Process Entire Dataset"])

def load_animal(root_dir):
    animal_list = []
    for animal_dir in os.listdir(root_dir):
        animal_path = os.path.join(root_dir, animal_dir)
        if os.path.isdir(animal_path):
            images = glob.glob(os.path.join(animal_path, "*.jpg"))
            for image_path in sorted(images):
                animal_list.append({
                    'animal_id': animal_dir,
                    'image_path': image_path
                })

    return pd.DataFrame(animal_list)

def read_and_unpack_image(file):
    
    try:
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

def generate_visualization(unpacked_image, data_segmentation, inline=False):
    """
    Generate visualizations for the unpacked image and prediction data.
    """
    thermal_image = unpacked_image['thermal']
    optical_image = unpacked_image['optical']
    head_mask = control.crop_mask_from_image(thermal_image, data_segmentation['masks'][0])
    head_bbox = control.crop_boxes_from_image(thermal_image, data_segmentation['boxes'][0])
    implant_mask = control.crop_mask_from_image(thermal_image, data_segmentation['masks'][1])
    implant_bbox = control.crop_boxes_from_image(thermal_image, data_segmentation['boxes'][1])
    donut_roi = control.calculate_donut_roi(control.crop_mask_from_image(thermal_image, data_segmentation['masks'][1]), thermal_image)

    dict_images = {
        'original': unpacked_image['original'],
        'optical': optical_image,
        'thermal': thermal_image,
        'grayscale': unpacked_image['gray'],
        'head_bbox': head_bbox,
        'head_mask': head_mask,
        'head_histogram': head_mask,
        'implant_bbox': implant_bbox,
        'implant_mask': implant_mask,
        'implant_histogram': implant_mask,  
        'donut_roi': donut_roi,
        'donut_roi_histogram': donut_roi,
        'temperature_profile_y': implant_mask,
        'temperature_profile_x': implant_mask,
    }

    if inline:
        nrows = 1
        ncols = len(dict_images)
        figsize = (ncols * 3, 3)
    else:
        nrows = 4
        ncols = 4
        figsize = (ncols * 3, 9)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (key, value) in enumerate(dict_images.items()):
        if i >= nrows * ncols: break
        ax = axes[i]
        cmap = 'inferno' if key is not 'original' or 'optical' else 'gray'
        if 'histogram' in key:
            ax.hist(value[value!=0], bins=30, alpha=0.7)
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {key}')
        elif 'temperature_profile_y' in key:
            for x in range(value.shape[1]):
                column = value[:, x]
                ax.plot(range(len(column)), column, color='blue', alpha=0.7)
        elif 'temperature_profile_x' in key:
            for i, line in enumerate(value): 
                ax.plot(range(len(line)), line, color='blue', alpha=0.7)
        else:    
            ax.imshow(value, cmap=cmap)
        ax.set_title(key)
    
    for j in range(i + 1, nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()
    return fig

with tab1:

    uploaded_file = st.file_uploader("Upload a image", 
                                     type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        
        unpacked_image = read_and_unpack_image(uploaded_file)
    
        with st.spinner(f'Sending {uploaded_file.name} to server ...'): data_segmentation = server_config.predict(unpacked_image['gray'])

        if data_segmentation is None:
            st.error("Error processing the image. Please check the file format and try again.")
            st.stop()
        
        st.markdown('---')
        st.caption(f"General Statistics")
        
        dataframe = pd.DataFrame(control.thermal_stats(np.array(unpacked_image['thermal']), unpacked_image['metadata']))
        st.dataframe(dataframe, use_container_width=True)
        
        st.markdown('---')
        st.caption(f"Visualizations of {uploaded_file.name}")

        st.pyplot(generate_visualization(unpacked_image, data_segmentation, inline=False)) 

with tab2:

    uploaded_files = st.file_uploader("Upload all files from a folder (select multiple)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    df_info = []

    for file in uploaded_files:

        st.write(f'Processing {file.name}')

        unpacked_image = read_and_unpack_image(file)

        with st.spinner(f'Sending {file.name} to server ...'): data_segmentation = server_config.predict(unpacked_image['gray'])

        if data_segmentation is None:
            st.error(f"Error processing {file.name}. Please check the file format and try again.")
            continue
        
        img_obj = ProcessedImage(original_image=unpacked_image['original'],
                                 optical_image=unpacked_image['optical'],
                                 thermal_matrix=unpacked_image['thermal'],
                                 grayscale=unpacked_image['gray'],
                                 animal_id="Sheep", 
                                 metadata=unpacked_image['metadata'],
                                 masks=data_segmentation['masks'],
                                 boxes=data_segmentation['boxes'],
                                 classes=data_segmentation['classes'])
              
        st.pyplot(generate_visualization(unpacked_image, data_segmentation, inline=True))

        st.button(f"Save {file.name} to database",
                  on_click=lambda: session.add(img_obj) or session.commit()) 

        df_info.append(img_obj.get_thermal_stats())
    
    all_info = session.query(ProcessedImage).count()
    st.write(f"Total images processed: {all_info}")
    

    if len(df_info) != 0:
        st.markdown('---')

        st.subheader("General Statistics")

        df_total = pd.concat(df_info, ignore_index=True)
        st.dataframe(df_total, use_container_width=True)
    
        st.markdown("#### Head")
        st.write('EM CONSTRUÇÃO')

        st.markdown("#### Implant")
        st.write('EM CONSTRUÇÃO')

with tab3:

    root_dir = st.text_input("Enter the root directory of the dataset", value="/Users/blb/Documents/DOUTORADO/FEMALES") # Adjust this path as needed

    if st.button("Load Dataset"):
        if os.path.exists(root_dir):
            animal_df = load_animal(root_dir)
            st.write(f"Found {len(animal_df)} animals in the dataset.")
            st.dataframe(animal_df, use_container_width=True)
        else:
            st.error("Directory does not exist. Please check the path and try again.")
