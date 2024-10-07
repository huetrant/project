import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

@st.cache_data
def grabcut_process(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)  # Giảm số lần lặp

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result

def resize_image(image, target_width):
    height, width, _ = image.shape
    if width > target_width:
        aspect_ratio = height / width
        target_height = int(target_width * aspect_ratio)
        resized_image = cv2.resize(image, (target_width, target_height))
        return resized_image
    return image

# Tạo giao diện Streamlit
st.title("GrabCut Algorithm")
st.markdown("""
- Add an image
- Draw a rectangle around the area to extract
- Click on the "Extract foreground" button
""")

# Tải ảnh
uploaded_file = st.file_uploader("Add Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Đọc ảnh và chuyển thành mảng numpy
    image = np.array(Image.open(uploaded_file))
    
    # Co giãn ảnh để vừa với chiều rộng mong muốn (ví dụ 700px)
    target_width = 700
    resized_image = resize_image(image, target_width)

    # Hiển thị canvas cho người dùng vẽ hình chữ nhật
    st.write("Draw a rectangle on the image")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        background_image=Image.fromarray(resized_image),
        update_streamlit=True,
        height=resized_image.shape[0],
        width=resized_image.shape[1],
        drawing_mode="rect",
        key="canvas",
    )

    if canvas_result.json_data is not None:
        # Chỉ xử lý GrabCut khi người dùng nhấn nút
        if st.button('Extract foreground'):
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    # Lấy tọa độ của hình chữ nhật mà người dùng đã vẽ
                    left = int(obj["left"])
                    top = int(obj["top"])
                    width = int(obj["width"])
                    height = int(obj["height"])

                    # Chuyển đổi tọa độ hình chữ nhật về ảnh gốc
                    scale_x = image.shape[1] / resized_image.shape[1]
                    scale_y = image.shape[0] / resized_image.shape[0]

                    # Chuyển thành tuple để sử dụng cho GrabCut
                    rect = (int(left * scale_x), int(top * scale_y), int((left + width) * scale_x), int((top + height) * scale_y))

                    result = grabcut_process(image, rect)
                    st.image(result, caption="Extracted Image", use_column_width=True)

                    # Nút tải ảnh kết quả
                    result_image = Image.fromarray(result)
                    buffered = BytesIO()
                    result_image.save(buffered, format="PNG")
                    st.download_button("Download Result", data=buffered.getvalue(), file_name="result.png", mime="image/png")
