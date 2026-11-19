import os
import io
import urllib.parse
import requests as rq
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
hf_api_url = "https://router.huggingface.co/hf-interface/models/black-forest-labs/FLUX.1-schnell"
hf_token = os.getenv("HF_TOKEN")

def generate_image_hf(prompt):
    try:
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed=43"

        response = rq.get(url, timeout=45)

        if response.status_code == 200:
            return response.content, None
        else:
            return None, f"сервис временно недоступен ({response.status_code})"
    except rq.exceptions.Timeout:
        return None, "timeout"
    except Exception as e:
        return None, f"ошибка подключения: {e}"

st.set_page_config(page_title="ai creative studio", page_icon="📸", layout="wide")

st.title("📸 ai creative studio & editor")
st.write("опиши какое изображение хочешь получить")

if "ai_image_bytes" not in st.session_state:
    st.session_state.ai_image_bytes = None

prompt = st.text_area("описание изображения (промпт):")

if st.button("сгенерировать изображение") and prompt:
    with st.spinner("ai создает изображение..."):
        image_bytes, error = generate_image_hf(prompt)

    if error:
        st.error(error)
    elif image_bytes:
        st.session_state.ai_image_bytes = image_bytes

if st.session_state.ai_image_bytes is not None:
    st.divider()
    st.subheader("редактирование сгенерированного фото")

    image = Image.open(io.BytesIO(st.session_state.ai_image_bytes))
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("сгенерированное ai фото")
        st.image(img_array, use_container_width=True)

    st.sidebar.header("настройка фильтров")
    filter_option = st.sidebar.selectbox("выберите фильтр: ",
                                       ["оригинал", "черно-белый",
                                        "размытие", "яркость",
                                        "контраст", "инверсия"])

    processed_img = img_array.copy()

    if filter_option == "черно-белый":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif filter_option == "размытие":
        processed_img = cv2.GaussianBlur(img_array, (15, 15), 0)
    elif filter_option == "яркость":
        brightness_val = st.sidebar.slider("уровень яркости", -100, 100, 0)
        processed_img = cv2.convertScaleAbs(img_array, alpha=1.0, beta=brightness_val)
    elif filter_option == "контраст":
        contrast_val = st.sidebar.slider("уровень контраста", 0.5, 3.0, 1.0, step=0.1)
        processed_img = cv2.convertScaleAbs(img_array, alpha=contrast_val, beta=0)
    elif filter_option == "инверсия":
        processed_img = 255 - img_array

    with col2:
        st.subheader("обработанная фотография")
        st.image(processed_img, use_container_width=True)

        result_image = Image.fromarray(processed_img)
        buf = io.BytesIO()
        result_image.save(buf, format="jpeg")
        byte_im = buf.getvalue()

        st.download_button(
            label="скачать",
            data=byte_im,
            file_name="smart_edit.jpeg",
            mime="image/jpeg"
        )