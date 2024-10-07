import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from io import BytesIO

@st.cache_data
def load_and_resize_image(uploaded_file, target_width):
    image = Image.open(uploaded_file)
    aspect_ratio = image.size[1] / image.size[0]
    target_height = int(target_width * aspect_ratio)
    resized_image = image.resize((target_width, target_height), Image.ANTIALIAS)
    return np.array(resized_image)

@st.cache_data
def grabcut_process(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result

st.title("GrabCut Algorithm")
st.markdown("""
- Add an image
- Draw a rectangle around the area to extract
- Click on the "Extract foreground" button
""")

uploaded_file = st.file_uploader("Add Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    resolution_option = st.selectbox("Choose image resolution", ["Low", "Medium", "High"])
    target_width = 300 if resolution_option == "Low" else 700 if resolution_option == "Medium" else 1400

    image = load_and_resize_image(uploaded_file, target_width)

    st.write("Draw a rectangle on the image")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=Image.fromarray(image),
        update_streamlit=True,
        height=image.shape[0],
        width=image.shape[1],
        drawing_mode="rect",
        key="canvas",
    )

    if canvas_result.json_data is not None:
        if st.button('Extract foreground'):
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    left = int(obj["left"])
                    top = int(obj["top"])
                    width = int(obj["width"])
                    height = int(obj["height"])

                    scale_x = image.shape[1] / image.shape[1]
                    scale_y = image.shape[0] / image.shape[0]

                    rect = (int(left * scale_x), int(top * scale_y), int((left + width) * scale_x), int((top + height) * scale_y))

                    with st.spinner("Processing..."):
                        result = grabcut_process(image, rect)
                        st.image(result, caption="Extracted Image", use_column_width=True)

                        # Nút tải ảnh kết quả
                        result_image = Image.fromarray(result)
                        buffered = BytesIO()
                        result_image.save(buffered, format="PNG")
                        st.download_button("Download Result", data=buffered.getvalue(), file_name="result.png", mime="image/png")
