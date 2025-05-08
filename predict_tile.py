import tensorflow as tf
import numpy as np
import cv2
import os
import json

image_path = "test_tile.png"
model_path = "mahjong_model"
class_indices_path = "class_indices.json"
img_height, img_width = 100, 100

model = tf.keras.models.load_model(model_path)

with open(class_indices_path, "r", encoding="utf-8") as f:
    class_indices = json.load(f)

index_to_label = {v: k for k, v in class_indices.items()}

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (img_width, img_height))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
predicted_index = int(np.argmax(pred))
predicted_label = index_to_label[predicted_index]
confidence = float(np.max(pred)) * 100

print(f"→ この牌は「{predicted_label}」であると{confidence:.1f}%の確信を持って予測しました。")

print("予測スコア:", pred)
print("予測されたインデックス:", predicted_index)
print("対応するラベル:", predicted_label)