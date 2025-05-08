import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
import json

csv_path = 'augmented_tiles_opencv/labels.csv'
base_dir = 'augmented_tiles_opencv'
img_height, img_width = 100, 100
batch_size = 32
epochs = 10

df = pd.read_csv(csv_path)
df = df.dropna(subset=['label', 'filename'])
df['filepath'] = df.apply(lambda row: os.path.join(base_dir, row['label'], row['filename']), axis=1)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.7, 1.3],
    zoom_range=0.1,
    horizontal_flip=True,
    )
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filepath', y_col='label',
    target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df, x_col='filepath', y_col='label',
    target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical'
)

num_classes = len(train_generator.class_indices)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    verbose=2
)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train_generator.class_indices, f, ensure_ascii=False, indent=4)

model.save("mahjong_model", save_format='tf')
print("モデルを保存しました。")

plt.show()