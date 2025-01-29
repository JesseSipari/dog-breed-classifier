import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Datasetin asetukset
TRAIN_DIR = "train"
TEST_DIR = "test"
LABELS_CSV = "labels.csv"

# 1. LUETAAN LABELIT CSV:STÄ JA MUODOSTETAAN POLUT KUVILLE
labels_df = pd.read_csv(LABELS_CSV)
labels_df["file_path"] = labels_df["id"].apply(lambda x: os.path.join(TRAIN_DIR, x + ".jpg"))

# Luodaan sanakirja breed -> numero
unique_breeds = labels_df["breed"].unique()
breed_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}
idx_to_breed = {v: k for k, v in breed_to_idx.items()}
labels_df["label_idx"] = labels_df["breed"].map(breed_to_idx)

# Tallennetaan idx_to_breed JSON-tiedostoon
with open("idx_to_breed.json", "w") as f:
    json.dump(idx_to_breed, f)

# Jaetaan train- ja validaatiosetteihin
train_df, val_df = train_test_split(
    labels_df,
    test_size=0.2,
    stratify=labels_df["breed"],
    random_state=42
)

# Data augmentaatio
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

IMG_SIZE = (299, 299)
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

def load_image_and_label(file_path, label_idx):
    image_raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label_idx

def augment(image, label):
    return data_augmentation(image), label

def create_dataset(df, shuffle=True, do_augment=False):
    file_paths = df["file_path"].values
    labels = df["label_idx"].values
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(load_image_and_label, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    if do_augment:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = create_dataset(train_df, shuffle=True, do_augment=True)
val_ds = create_dataset(val_df, shuffle=False, do_augment=False)

# Mallin rakentaminen
from tensorflow.keras.applications import InceptionV3

base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation="relu")(x)
predictions = layers.Dense(len(unique_breeds), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Ensimmäinen koulutusvaihe
EPOCHS_1 = 6
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_1)

# Fine-tuning
fine_tune_at = 249
for layer in model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in model.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

EPOCHS_2 = 12
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_2)

# Parannettu tallennuslogiikka
MODEL_PATH = "dog_breed_inception.keras"

if os.path.exists(MODEL_PATH):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_path = f"dog_breed_inception_{timestamp}.keras"
    model.save(new_model_path)
    print(f"Tiedosto '{MODEL_PATH}' on jo olemassa. Malli tallennettu tiedostoon: {new_model_path}")
else:
    model.save(MODEL_PATH)
    print(f"Malli tallennettu tiedostoon: {MODEL_PATH}")

# Testauksen valmistelu
test_files = os.listdir(TEST_DIR)
test_paths = [os.path.join(TEST_DIR, f) for f in test_files if f.endswith(".jpg")]

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

def load_test_image(file_path):
    image_raw = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image

test_ds = test_ds.map(load_test_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

predictions = model.predict(test_ds)
predicted_classes = np.argmax(predictions, axis=1)

# Tulosta esimerkki ennusteista
for i in range(10):
    print(test_files[i], "-> ennustettu:", idx_to_breed[str(predicted_classes[i])])
