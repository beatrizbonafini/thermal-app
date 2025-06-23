import numpy as np
import cv2
import scipy.stats as stats
import flyr
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ExifTags
import pandas as pd
from io import BytesIO
import streamlit as st
import tempfile
import shutil


def calculate_mean(roi): return '{:.2f}'.format(np.mean(roi))

def calculate_max(roi): return '{:.2f}'.format(np.max(roi))

def calculate_min(roi): return '{:.2f}'.format(np.min(roi))

def calculate_std(roi): return '{:.2f}'.format(np.std(roi))

def calculate_median(roi): return '{:.2f}'.format(np.median(roi))

def calculate_var(roi): return '{:.2f}'.format(np.var(roi))

def calculate_percentil(roi, percentil): return '{:.2f}'.format(np.percentile(roi, percentil))

def calculate_entropy(roi): 
    
    histograma = np.histogram(roi[roi!=0], bins=256, range=(0, 256))[0]
    histograma = histograma / histograma.sum()

    #plt.hist(roi[roi!=0])
    #plt.show()
    return '{:.2f}'.format(stats.entropy(histograma))

def calculate_gradient(roi):

    gradient_x = np.gradient(roi, axis=0)
    gradient_y = np.gradient(roi, axis=1)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    return '{:.2f}'.format(np.mean(gradient_magnitude))

def calculate_heating_rate(roit1, roit2): return '{:.2f}'.format(np.mean(roit1) - np.mean(roit2))

def calculate_spatial_assimetry(roi): 

    colunas = roi.shape[1]
    metade = colunas // 2

    matriz_esquerda = roi[:, :metade]
    matriz_direita = roi[:, -metade:]

    T_L = np.mean(matriz_esquerda)
    T_R = np.mean(matriz_direita)

    return '{:.2f}'.format(abs(T_R - T_L) / (T_R + T_L))

def get_thermal_matrix(flir_path):  return flyr.unpack(flir_path).celsius

def calculate_donut_roi(mask):
    
    th, threshed = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    dilated = cv2.dilate(threshed, kernel, iterations=2)
    eroded = cv2.erode(threshed, kernel, iterations=1)
    eroded[eroded == 255] = -255
    morph_result = eroded + dilated
    morph_result[morph_result == 255] = 1

    return morph_result

def calculate_just_cicle_roi(mask):
    th, threshed = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    threshed[threshed == 255] = 1
    return threshed

def get_roi_box(thermal_matrix):
    
    coords = np.column_stack(np.where(thermal_matrix > 0))
    if len(coords) == 0: return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped_image = thermal_matrix[y_min:y_max+1, x_min:x_max+1]
    return cropped_image

def resize_annotation(annotation, image_shape_original, image_shape_resized):
    
    for shape in annotation['shapes']:
        for point in shape['points']:
            point[0] = point[0] * image_shape_resized[1] / image_shape_original[1]
            point[1] = point[1] * image_shape_resized[0] / image_shape_original[0]

    return annotation

def draw_gradient(roi):
    
    gradient_x = np.gradient(roi, axis=0)
    gradient_y = np.gradient(roi, axis=1)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return gradient_x, gradient_y, gradient_magnitude

def min_max_normalize_array(max, min, valor): return '{:.2f}'.format((valor - min) / (max - min))

def thermal_stats(thermal_matrix, exif_data=None):
    
    info = {}
    
    if exif_data is not None:
        for tag_id, valor in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == 'MakerNote' or tag == 'ComponentsConfiguration': continue
            info[tag] = valor
    else:
        info['MakerNote'] = 'No EXIF data found'

    stats = {
        'max': [calculate_max(thermal_matrix)],
        'min': [calculate_min(thermal_matrix)],
        'mean': [calculate_mean(thermal_matrix)],
        'std':[ calculate_std(thermal_matrix)],
        'median': [calculate_median(thermal_matrix)],
        'var': [calculate_var(thermal_matrix)],
        'percentil_5': [calculate_percentil(thermal_matrix, 5)],
        'percentil_95': [calculate_percentil(thermal_matrix, 95)],
    } | info
    return pd.DataFrame(stats)

def image_histogram(thermal_image):

    fig, ax = plt.subplots()
    data = np.unique(thermal_image[thermal_image!=0])
    sns.histplot(data, bins=60, kde=True)

    min_val = data.min()
    max_val = data.max()
    mean_val = data.mean()
    th95 = np.percentile(data, 95)
    th5 = np.percentile(data, 5)

    ax.axvline(min_val, color='black', linestyle='--', label=f'Min: {min_val:.2f}')
    ax.axvline(max_val, color='black', linestyle='--', label=f'Max: {max_val:.2f}')
    ax.axvline(mean_val, color='black', linestyle='--', label=f'Mean: {max_val:.2f}')
    ax.axvline(th95, color='black', linestyle='--', label=f'Mean: {max_val:.2f}')
    ax.axvline(th5, color='black', linestyle='--', label=f'Mean: {max_val:.2f}')

    ax.text(min_val, plt.ylim()[1]*0.98, f'-Min', color='black', ha='left', va='top')
    ax.text(mean_val, plt.ylim()[1]*0.98, f'-Mean', color='black', ha='left', va='top')
    ax.text(max_val, plt.ylim()[1]*0.98, f'-Max', color='black', ha='left', va='top')
    ax.text(th95, plt.ylim()[1]*0.98, f'95th-', color='black', ha='right', va='top')
    ax.text(th5, plt.ylim()[1]*0.98, f'-5th', color='black', ha='left', va='top')


    ax.set_title('Thermal distribution at the implant boundary')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Frequency (pixels)')
    #plt.legend()
    return fig

def get_rgb(flir_path): return flyr.unpack(flir_path).optical 

# preciso ter um método que retorne em escala de cinza a imagem térmica
def get_thermal_gray(flir_path): 
    palletes = ["grayscale"]
    thermogram = flyr.unpack(flir_path)
    for p in palletes:
        render = thermogram.render_pil(
            min_v=20,
            max_v=40,
            unit='celsius',
            palette=p,
        )
        render.save("gray_image.jpg")

def crop_boxes_from_image(thermal_matrix, bbox, 
                          round_mode: str = "floor"):
    
    H, W = thermal_matrix.shape 

    x_min, y_min, x_max, y_max = bbox
    
    round_fn = {"floor": np.floor, "round": np.round, "ceil": np.ceil}[round_mode]

    x_min = int(max(0, min(W,   round_fn(x_min))))
    y_min = int(max(0, min(H,   round_fn(y_min))))
    x_max = int(max(0, min(W,   round_fn(x_max))))
    y_max = int(max(0, min(H,   round_fn(y_max))))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Bounding box inválido ou zerado.")
    
    cropped = thermal_matrix[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        raise ValueError("Bounding box resultou em uma matriz vazia.")
    
    return cropped

def crop_mask_from_image(thermal_matrix: np.ndarray,
                               mask: np.ndarray,
                               fill_value=np.nan) -> np.ndarray:
    
    if thermal_matrix.shape != np.array(mask).shape:
        raise ValueError("A máscara e a matriz devem ter o mesmo shape.")
    
    masked_matrix = np.where(np.array(mask).astype(bool), thermal_matrix, fill_value)
    return masked_matrix

def matplotlib_figure_to_image(fig):
    """Converts a matplotlib figure to a PIL image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

def calculate_donut_roi(mask, thermal_matrix):
    
    th, threshed = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(threshed, kernel, iterations=2)
    eroded = cv2.erode(threshed, kernel, iterations=1).astype(np.int16)
    eroded[eroded == 255] = -255
    morph_result = eroded + dilated
    morph_result = (morph_result == 255).astype(np.uint8) 

    roi = morph_result * thermal_matrix
    
    return roi


def unpack_from_bytes(file_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            thermogram = flyr.unpack(tmp_path)
            celsius = thermogram.celsius
            optical = thermogram.optical
            grayscale = thermogram.render_pil(
                min_v=20,
                max_v=40,
                unit='celsius',
                palette='grayscale'
            )
            
        finally:
            shutil.os.remove(tmp_path)
        
        return (celsius, optical, grayscale)
