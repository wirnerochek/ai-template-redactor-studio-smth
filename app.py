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

formats = {
    "16:9 — Баннер YouTube (1280x720)": (1280, 720),
    "16:9 — Экран ПК / Монитор (1366x768)": (1366, 768),
    "16:9 — Телевизор TV (1920x1080)": (1920, 1080),
    "4:3 — Планшет (1200x900)": (1200, 900),
    "9:16 — Смартфон / Телефон (720x1280)": (720, 1280),
    "1:1 — Аватарка (1024x1024)": (1024, 1024),
    "1:1 — Смарт-часы / Гаджет (800x800)": (800, 800),
}


def generate_image_hf(prompt, width, height, token):
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true&seed=43"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = rq.get(url, headers=headers, timeout=45)
    if response.status_code != 200:
        raise RuntimeError(f"Сервис вернул статус {response.status_code}")
    return response.content


def apply_template(img_array, target_w, target_h):
    pil_img = Image.fromarray(img_array).convert("RGB")
    src_w, src_h = pil_img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    return np.array(cropped)


st.set_page_config(page_title="AI Template & Edit Studio", layout="wide")

st.title("AI Template & Edit Studio")
st.write("Создавай картинки под нужный формат, применяй фильтры и скачивай готовый шаблон")

if "ai_image_bytes" not in st.session_state:
    st.session_state.ai_image_bytes = None

col_prompt, col_tpl = st.columns([2, 1])

with col_prompt:
    prompt = st.text_area("Описание изображения (только на русском языке):")

with col_tpl:
    selected_format = st.selectbox("Выбери формат:", list(formats.keys()))

target_w, target_h = formats[selected_format]

if st.button("Сгенерировать изображение"):
    if not prompt.strip():
        st.warning("Ты не ввел описание изображения — напиши промпт в поле выше и нажми кнопку снова")
    else:
        try:
            with st.spinner("AI генерирует изображение под выбранный формат..."):
                image_bytes = generate_image_hf(prompt.strip(), target_w, target_h, hf_token)
                st.session_state.ai_image_bytes = image_bytes
                st.success("Изображение готово — настрой фильтр в боковой панели и скачай готовый шаблон")
        except Exception as e:
            st.error(f"Не удалось получить изображение ({e}) — проверь подключение к сети и попробуй снова")

if st.session_state.ai_image_bytes is not None:
    st.divider()
    st.subheader(f"Формат: {selected_format}")

    image = Image.open(io.BytesIO(st.session_state.ai_image_bytes)).convert("RGB")
    img_array = np.array(image)

    st.sidebar.header("Настройка фильтров")
    filter_option = st.sidebar.selectbox(
        "Выберите фильтр:",
        ["Оригинал", "Черно-белый", "Размытие", "Яркость", "Контраст", "Инверсия"]
    )

    processed_img = img_array.copy()

    if filter_option == "Черно-белый":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    elif filter_option == "Размытие":
        processed_img = cv2.GaussianBlur(img_array, (15, 15), 0)
    elif filter_option == "Яркость":
        brightness_val = st.sidebar.slider("Уровень яркости", -100, 100, 0)
        processed_img = cv2.convertScaleAbs(img_array, alpha=1.0, beta=brightness_val)
    elif filter_option == "Контраст":
        contrast_val = st.sidebar.slider("Уровень контраста", 0.5, 3.0, 1.0, step=0.1)
        processed_img = cv2.convertScaleAbs(img_array, alpha=contrast_val, beta=0)
    elif filter_option == "Инверсия":
        processed_img = 255 - img_array

    final_img = apply_template(processed_img, target_w, target_h)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Исходник от AI")
        st.image(img_array, use_container_width=True)

    with col2:
        st.subheader("Готовый шаблон")
        st.image(final_img, use_container_width=True)

        result_image = Image.fromarray(final_img)
        buf = io.BytesIO()
        result_image.save(buf, format="jpeg")
        byte_im = buf.getvalue()

        st.download_button(
            label="Скачать шаблон",
            data=byte_im,
            file_name="template_result.jpeg",
            mime="image/jpeg"
        )
