import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


img = cv2.imread("Merge_Timex_BoaViagem.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

lower = np.array([40, 40, 40])
upper = np.array([150, 150, 150]) 

mask = cv2.inRange(img_rgb, lower, upper)
mask_inv = cv2.bitwise_not(mask)
img_filtered = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_inv)

gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

black_sum = np.sum(binary == 255, axis=1)
threshold = np.max(black_sum) * 0.75
lines_to_keep = black_sum >= threshold

filtered = np.zeros_like(binary)

for i, keep in enumerate(lines_to_keep):
    if keep:
        filtered[i, :] = binary[i, :]

edges = cv2.Canny(filtered, threshold1=50, threshold2=150)
output = np.zeros_like(edges)

for col in range(edges.shape[1]):
    col_data = edges[:, col]
    white_pixels = np.where(col_data == 255)[0]
    if len(white_pixels) > 0:
        last_white = white_pixels[-1]
        output[last_white, col] = 255

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
connected = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)

mask = connected == 255
color = [0, 0, 255]  # vermelho
img_overlay = img_rgb.copy()
img_overlay[mask] = color

fig, axs = plt.subplots(2, 4, figsize=(12, 6))

imgs = [img_rgb, img_filtered, gray, binary, filtered, output, connected, img_overlay]
titles = ['Imagem Original', 
          'Imagem sem intervalo RGB', 
          'Imagem em escala de cinza',
          'Imagem binarizada', 
          'Linhas com maior limiar de branco', 
          'Mantendo borda mais baixa', 
          'Operação de Fechamento', 'Resultado Final']

for ax, img, title in zip(axs.flat, imgs, titles):
    ax.imshow(img if len(img.shape) == 3 else img, cmap='gray' if len(img.shape) == 2 else None)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
