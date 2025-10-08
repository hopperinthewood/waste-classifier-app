import requests 
import tensorflow as tf 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
import numpy as np 
from PIL import Image 
import streamlit as st 
import os 

# ดาวน์โหลดโมเดลจาก Google Drive 
file_id = "13DhMqBA1YqjQKEA6F5UXIdcUq7_j9iZm" 
download_url = f"https://drive.google.com/uc?export=download&id={file_id}" 
response = requests.get(download_url) 
with open("model.h5", "wb") as f: 
    f.write(response.content) 
    
# โหลดโมเดลด้วย TensorFlow 
model = load_model("model.h5") 

# กำหนดชื่อคลาส (ปรับตาม dataset ที่ใช้เทรน) 
class_labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ตั้งค่า Streamlit 
st.set_page_config(page_title="Waste Classification App", layout="wide") 
st.markdown("<h1 style='color:#2c3e50; '>♻️ Waste Classification App</h1>", unsafe_allow_html=True) 
st.write("อัปโหลดภาพขยะเพื่อให้โมเดลจำแนกประเภทโดยใช้ Deep Learning") 

# Layout 2 คอลัมน์ 
col1, col2 = st.columns([1, 2]) 

with col1: 
    uploaded_file = st.file_uploader("📤 เลือกรูปภาพขยะ...", type=["jpg", "jpeg", "png"]) 
    
with col2: 
    st.markdown(""" 
    ### 📝 คำแนะนำ: 
    - รองรับไฟล์ .jpg, .jpeg, .png 
    - ภาพควรแสดงประเภทขยะชัดเจน เช่น plastic, paper, metal, glass 
    - โมเดลจะทำการจำแนกประเภทขยะและแสดงความมั่นใจ 
    """) 
    
# แสดงผลการจำแนก 
if uploaded_file is not None: 
    img = Image.open(uploaded_file) 
    st.image(img, caption="📷 ภาพที่อัปโหลด", use_container_width=True) 
    
    img = img.resize((128, 128)) 
    img_array = image.img_to_array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    
    prediction = model.predict(img_array) 
    predicted_class = class_labels[np.argmax(prediction)] 
    confidence = np.max(prediction) 
    
    st.markdown(f"<h3 style='color:#27ae60;'>🧠 ประเภทขยะที่ทำนายได้: <strong>{predicted_class}</strong></h3>", unsafe_allow_html=True) 
    st.markdown(f"<h3 style='color:#2980b9;'>📊 ความมั่นใจ: <strong>{confidence:.2f}</strong></h3>", unsafe_allow_html=True)

