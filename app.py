import streamlit as st
import cv2
import numpy as np
from sklearn.cluster import KMeans
import tempfile
import os
from PIL import Image

st.set_page_config(page_title="Color Segmenter", layout="wide")
st.title("🎨 Color Segmenter for Illustrations")

st.sidebar.header("Options")
n_colors = st.sidebar.slider("Max Number of Colors", 2, 20, 6)
simplify_segments = st.sidebar.slider("Segment Simplification", 0, 10, 2)
merge_threshold = st.sidebar.slider("Merge Similar Colors", 0, 100, 20)

uploaded_files = st.sidebar.file_uploader("Upload Illustration(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

def extract_colors(image, n_colors):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = np.array(kmeans.cluster_centers_, dtype='uint8')
    labels = kmeans.labels_.reshape(image.shape[:2])
    return colors, labels

def create_masked_images(image, labels, colors, simplify):
    results = []
    for i, color in enumerate(colors):
        mask = (labels == i).astype(np.uint8) * 255
        if simplify > 0:
            kernel = np.ones((simplify, simplify), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        colored = cv2.bitwise_and(image, image, mask=mask)
        results.append((i + 1, colored))
    return results

def draw_color_legend(image, colors):
    legend_height = 50
    legend = np.ones((legend_height, image.shape[1], 3), dtype=np.uint8) * 255
    step = image.shape[1] // len(colors)
    for i, color in enumerate(colors):
        x = i * step
        legend[:, x:x + step] = color
        cv2.putText(legend, str(i + 1), (x + 5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return np.vstack((image, legend))

def process_image(file):
    image = Image.open(file).convert('RGB')
    image_np = np.array(image)
    colors, labels = extract_colors(image_np, n_colors)
    cutouts = create_masked_images(image_np, labels, colors, simplify_segments)
    numbered_image = draw_color_legend(image_np.copy(), colors)
    return numbered_image, cutouts

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processed: {uploaded_file.name}")
        numbered_img, cutouts = process_image(uploaded_file)

        st.image(numbered_img, caption="Numbered Colors", use_column_width=True)

        st.markdown("### Color Cutouts")
        cols = st.columns(3)
        for idx, (color_num, cutout) in enumerate(cutouts):
            with cols[idx % 3]:
                st.image(cutout, caption=f"Color #{color_num}")
else:
    st.info("Upload one or more illustration images to begin.")
