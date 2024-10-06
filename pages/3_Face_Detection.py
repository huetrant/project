import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Face Detection với Haar Cascade")
st.subheader("1. Trainning")
st.subheader("1.1. Data")
data_path = "./images/output/data.png"  
st.image(data_path, caption='Một số mẫu dữ liệu', width=1000)

st.subheader("1.2. Tham số")
st.markdown("""
- numPos <800>: Số lượng mẫu dương (positive samples) được sử dụng trong quá trình huấn luyện. 
- numNeg <400>: Số lượng mẫu âm (negative samples) được sử dụng trong quá trình huấn luyện. Số lượng này phụ thuộc vào các ảnh trong tệp background.
- numStages <5>: Số lượng giai đoạn (stages) trong quá trình huấn luyện cascade. Mỗi giai đoạn sẽ thêm một lớp phân loại mới vào mô hình.
- numThreads <4>:  Số lượng luồng tối đa mà quá trình huấn luyện có thể sử dụng để tăng tốc độ huấn luyện.
""")

st.subheader("2. Testing")
st.subheader("2.1. Groundtruth từ model Haar cascadee (OpenCV)")
gt_path = "./images/output/gt_opcv.png"  
st.image(gt_path, caption='Minh họa phát hiện khuôn mặt từ Haar cascadee (OpenCV) ', width=1000)