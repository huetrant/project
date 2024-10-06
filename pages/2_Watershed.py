import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Watershed Segmentation")

st.header("Phân đoạn ký tự biển số xe với Watershedd")
st.subheader("1. Trainning")
st.subheader("1.1. Trainning imgaes")


# Đường dẫn đến hai hình ảnh
image_path1 = "./images/watershed/1xemay356.jpg"  
image_path2 = "./images/watershed/1xemay358.jpg"

st.image(image_path1, caption='Biển số 1', width=700)
st.image(image_path2, caption='Biển số 2', width=700)

st.subheader("1.2. Thiết lập thí nghiệm")
st.markdown("<h3 style='font-size: 25px; color: #FFFFFF;'>Bộ tham số:</h3>", unsafe_allow_html=True)
st.markdown("""
- Kernel size: [3, 5, 7]
- Dist transform threshold: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
""")
st.markdown("<h3 style='font-size: 25px; color: #FFFFFF;'>Quá trình thí nghiệm: </h3>", unsafe_allow_html=True)

pipeline_path = "./images/output/popeline.png"
st.image(pipeline_path, caption='Pipeline', width=1000)

st.markdown("<h3 style='font-size: 25px; color: #FFFFFF;'>Độ đo: </h3>", unsafe_allow_html=True)

IOU_path = "./images/output/IoU.webp"
st.image(IOU_path, caption='Công thức tính IoU', width=700)

st.markdown("<h3 style='font-size: 25px; color: #FFFFFF;'>Kết quả: </h3>", unsafe_allow_html=True)

bd_path = "./images/output/bd.png"
st.image(bd_path, caption='Biểu đồ kết quả tính IoU theo bộ tham số', width=1000)

st.markdown("<h3 style='font-size: 25px; color: #FFFFFF;'>Minh họa: </h3>", unsafe_allow_html=True)

mask30 = "./images/output/1xemay356_mask5-03.png"
st.image(mask30, caption='Mask với Kernel size = 5, Threshold = 0.3', width=700)

mask60 = "./images/output/1xemay356_mask5-06.png"
st.image(mask60, caption='Mask với Kernel size = 5, Threshold = 0.5', width=700)

mask80 = "./images/output/1xemay356_mask5-08.png"
st.image(mask80, caption='Mask với Kernel size = 5, Threshold = 0.7', width=700)

st.subheader("2. Testing")
st.subheader("2.1. Testing imgaes")

# Đường dẫn đến hai hình ảnh
image_path3 = "./images/watershed/1xemay376.jpg"  
st.image(image_path3, caption='Biển số 1', width=700)

image_path4 = "./images/watershed/1xemay399.jpg"
st.image(image_path4, caption='Biển số 2', width=700)

st.subheader("2.2. Kết quả: ")
test1 = "./images/output/1xemay376_mask.png"  
st.image(test1, caption='Biển số 1 với IoU = 25.53', width=700)

test2 = "./images/output/1xemay399_mask.png"  
st.image(test2, caption='Biển số 2 với IoU = 28.45', width=700)