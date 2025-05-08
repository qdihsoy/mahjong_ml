# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import json
import os

model = tf.keras.models.load_model("mahjong_model")
with open("class_indices.json", "r", encoding="utf-8") as f:
    class_indices = json.load(f)
index_to_label = {v: k for k, v in class_indices.items()}

st.title("麻雀牌分類器")
uploaded_file = st.file_uploader("牌の画像をアップロードしてください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name

    st.image(temp_file_path, caption="アップロードされた画像", use_column_width=True)

    if st.button("この牌を判定"):
        img = cv2.imread(temp_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        index = np.argmax(pred)
        label = index_to_label[index]
        confidence = float(np.max(pred)) * 100

        st.success(f"この牌は「{label}」であると{confidence:.1f}%の確信を持って予測しました。")
