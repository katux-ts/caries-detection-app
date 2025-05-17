# импорт необходимых библиотек
import streamlit as st
from PIL import Image
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from io import BytesIO
from datetime import datetime

# reportlab для PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

# регистрация шрифта
pdfmetrics.registerFont(TTFont('DejaVu', 'C:/Users/Mi/vs_projects/caries_detection_app/fonts/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', 'C:/Users/Mi/vs_projects/caries_detection_app/fonts/DejaVuSans-Bold.ttf'))

# заголовок и описание приложения
st.markdown("<h2 style='text-align: center;'>Диагностика кариеса у детей от 1 до 4 лет</h2>", unsafe_allow_html=True)
st.markdown("Приложение предназначено для первичной диагностики состояния зубов у детей раннего возраста.")

# загрузка обученной модели
@st.cache_resource
def load_model():
    model = torch.load("final_model.pth", map_location=torch.device("cpu"), weights_only=False)
    model.eval()
    return model

model = load_model()

# трансформации
val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def predict(image_pil):
    image_np = np.array(image_pil)
    transformed = val_transforms(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][class_idx].item()
    return class_idx, confidence

def process_uploaded_file(uploaded_file):
    return Image.open(uploaded_file).convert("RGB")

caries_recommendations = [
    "избегайте сладких напитков и перекусов перед сном",
    "начните использовать фторсодержащую детскую зубную пасту",
    "чистите зубы ребёнку 2 раза в день",
    "не давайте сладости между приёмами пищи"
]

healthy_recommendations = [
    "продолжайте регулярный уход",
    "посещайте стоматолога раз в 6 месяцев",
    "используйте мягкую щётку и фторсодержащую пасту",
    "избегайте соков и сладких напитков перед сном",
    "чистите зубы ребёнку утром и вечером по 2 минуты"
]

step = st.session_state.get('step', 1)

# шаг 1
if step == 1:
    st.markdown("<h3>Шаг 1: Загрузите фото с фронтального ракурса</h3>", unsafe_allow_html=True)
    uploaded_file_1 = st.file_uploader("Фото фронтального ракурса", type=["jpg", "jpeg", "png", "heic"])
    st.markdown("**Как сделать правильное фото?**")
    st.markdown("1. Фото должно быть сделано при хорошем освещении (естественном или лампе сбоку).\n" 
                "2. Важно, чтобы изображение было четким, не размытым, и в кадре не было лишних предметов.\n" 
                "3. Голова ребёнка должна смотреть прямо, рот открыт, зубы хорошо видны, губы не мешают обзору.")
    st.markdown('<a href="/static/front_example.jpeg" target="_blank"> Пример фото 📷</a> ', unsafe_allow_html=True)

    if uploaded_file_1:
        image_1 = process_uploaded_file(uploaded_file_1)
        st.image(image_1, caption="Фронтальное фото", use_container_width=True)

        col1, col2, col3 = st.columns([6, 1, 1])
        with col3:
            if st.button("Далее"):
                st.session_state['image_1'] = image_1
                st.session_state['step'] = 2
                st.rerun()

# шаг 2
elif step == 2:
    st.markdown("<h3>Шаг 2: Загрузите фото верхней челюсти (необязательно)</h3>", unsafe_allow_html=True)
    uploaded_file_2 = st.file_uploader("Фото верхней челюсти", type=["jpg", "jpeg", "png", "heic"])

    st.markdown('<a href="/static/upper_example.jpg" target="_blank"> Пример фото 📷</a> ', unsafe_allow_html=True)

    if uploaded_file_2:
        image_2 = process_uploaded_file(uploaded_file_2)
        st.image(image_2, caption="Верхняя челюсть", use_container_width=True)
        st.session_state['image_2'] = image_2

    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("Назад"):
            st.session_state['step'] = 1
            st.rerun()
    with col3:
        if st.button("Далее"):
            st.session_state['step'] = 3
            st.rerun()

# шаг 3
elif step == 3:
    st.markdown("<h3>Шаг 3: Загрузите фото нижней челюсти (необязательно)</h3>", unsafe_allow_html=True)
    uploaded_file_3 = st.file_uploader("Фото нижней челюсти", type=["jpg", "jpeg", "png", "heic"])

    st.markdown('<a href="/static/lower_example.JPG" target="_blank"> Пример фото 📷</a> ', unsafe_allow_html=True)

    if uploaded_file_3:
        image_3 = process_uploaded_file(uploaded_file_3)
        st.image(image_3, caption="Нижняя челюсть", use_container_width=True)
        st.session_state['image_3'] = image_3

    col1, col2, col3 = st.columns([1, 4.5, 2])
    with col1:
        if st.button("Назад"):
            st.session_state['step'] = 2
            st.rerun()

    run_analysis = False
    with col3:
        if st.button("Результаты анализа"):
            run_analysis = True

    if run_analysis:
        results = []

        if 'image_1' in st.session_state:
            class_idx, conf = predict(st.session_state['image_1'])
            results.append(("Фронтальное фото", class_idx, conf))

        if 'image_2' in st.session_state:
            class_idx, conf = predict(st.session_state['image_2'])
            results.append(("Верхняя челюсть", class_idx, conf))

        if 'image_3' in st.session_state:
            class_idx, conf = predict(st.session_state['image_3'])
            results.append(("Нижняя челюсть", class_idx, conf))

        final_label = "Кариес обнаружен" if any(class_idx == 0 for _, class_idx, _ in results) else "Кариес отсутствует"
        avg_confidence = round(np.mean([conf for _, _, conf in results]) * 100)

        if final_label == "Кариес обнаружен":
            st.markdown(f"### Итог: {final_label}")
            st.markdown(f"Вероятность наличия кариеса составляет {avg_confidence}%.")
            selected = random.sample(caries_recommendations, k=1)
            recommendations_text = ", ".join(selected) + "."
            st.markdown(f"<b>Рекомендации:</b> обратитесь к детскому стоматологу в ближайшие дни, {recommendations_text}", unsafe_allow_html=True)
        else:
            st.markdown(f"### Итог: {final_label}")
            st.markdown(f"Кариес отсутствует с вероятностью {avg_confidence}%.")
            selected = random.sample(healthy_recommendations, k=2)
            recommendations_text = ", ".join(selected) + "."
            st.markdown(f"<b>Рекомендации:</b> {recommendations_text}", unsafe_allow_html=True)

        # формирование PDF-отчета
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=A4)
        width, height = A4
        y = height - 50

        c.setFont("DejaVu-Bold", 16)
        c.drawCentredString(width / 2, y, "Отчет по диагностике кариеса")
        c.setFont("DejaVu", 12)
        y -= 30
        c.drawString(50, y, f"Дата и время анализа: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        y -= 20
        c.drawString(50, y, f"Итог: {final_label}")
        y -= 20
        c.drawString(50, y, f"Уверенность модели: {avg_confidence}%")
        y -= 30

        c.drawString(50, y, "Рекомендации:")
        y -= 20
        for line in recommendations_text.split(","):
            if final_label == "Кариес отсутствует":
                c.drawString(60, y, f"- {line.strip()}")
                y -= 20
            else:
                c.drawString(60,y, "- обратитесь к детскому стоматологу в ближайшие дни")
                y -= 20
                c.drawString(60, y, f"- {line.strip()}")
                y -= 20
                

        c.setFillColorRGB(1, 0, 0)
        c.drawString(50, y, "⚠️ Результаты анализа не являются медицинским диагнозом.")
        c.setFillColorRGB(0, 0, 0)

        image_map = {
            "Фронтальное фото": st.session_state.get('image_1'),
            "Верхняя челюсть": st.session_state.get('image_2'),
            "Нижняя челюсть": st.session_state.get('image_3')
        }

        y -= 30  # немного отступа перед фото
        c.setFont("DejaVu-Bold", 12)
        c.drawString(50, y, "Загруженные изображения для анализа")
        c.setFont("DejaVu", 12)
        y -= 20

        for label, img in image_map.items():
            if img:
                c.setFont("DejaVu", 12)
                c.drawString(60, y, f"{label}:")

                img_buffer = BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                image = ImageReader(img_buffer)
                img_width = 250  
                img_height = 150
                c.drawImage(image, 60, y - img_height, width=img_width, height=img_height, preserveAspectRatio=True, mask='auto')
                y -= (img_height + 20)


        c.save()
        pdf_buffer.seek(0)

        st.download_button(
            label="📄 Скачать PDF-отчет",
            data=pdf_buffer,
            file_name="diagnosis_report.pdf",
            mime="application/pdf"
        )

# предупреждение
st.warning("⚠️ Результаты анализа не являются медицинским диагнозом. Для точной оценки обратитесь к специалисту.")
