# plant_disease_complete_optimized.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# CONFIGURAÇÕES
# -------------------------------
base_dir = r"C:\Users\julio\OneDrive\Área de Trabalho\tcc\plant_village_dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

img_size = (224, 224)
batch_size = 32
epochs = 30  # mais épocas para melhor treino

# -------------------------------
# DATA AUGMENTATION
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = val_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)
print(f"Número de classes: {num_classes}")
class_labels = list(train_data.class_indices.keys())

# -------------------------------
# MODELO
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# CALLBACKS
# -------------------------------
model_path = os.path.join(base_dir, "plant_disease_model_best.h5")
checkpoint = ModelCheckpoint(
    model_path, monitor='val_accuracy', save_best_only=True, verbose=1
)
earlystop = EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1
)

# -------------------------------
# TREINAMENTO
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint, earlystop]
)

# -------------------------------
# AVALIAR NO TESTE
# -------------------------------
loss, accuracy = model.evaluate(test_data)
print(f"Acurácia no teste: {accuracy*100:.2f}%")

# -------------------------------
# PLOTAR GRÁFICOS
# -------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Acurácia')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.show()

# -------------------------------
# TESTAR NOVAS IMAGENS
# -------------------------------
def predict_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    print(f"Imagem: {os.path.basename(image_path)} -> Classe prevista: {class_labels[class_idx]}, Probabilidade: {pred[0][class_idx]:.2f}")

# Exemplo interativo
if __name__ == "__main__":
    import glob
    new_images_dir = r"C:\Users\julio\OneDrive\Área de Trabalho\tcc\novas_imagens"
    image_files = glob.glob(os.path.join(new_images_dir, "*.*"))
    
    print(f"Testando {len(image_files)} novas imagens...\n")
    for img_path in image_files:
        predict_image(img_path)
