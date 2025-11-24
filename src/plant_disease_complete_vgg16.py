# plant_disease_complete_vgg16.py (Implementação de Transfer Learning)

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Importação da arquitetura VGG16
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURAÇÕES
# -------------------------------
base_dir = r"C:\Users\julio\OneDrive\Área de Trabalho\tcc\plant_village_dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# VGG16 usa 224x224 como tamanho ideal, o que já estava sendo usado
img_size = (224, 224) 
batch_size = 32
epochs = 20 # Reduzir épocas, pois o TL converge mais rápido
RANDOM_SEED = 42 # Seed para reprodutibilidade

# -------------------------------
# DATA AUGMENTATION
# -------------------------------
# Aumento de dados (Data Augmentation) permanece o mesmo
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

# OTIMIZAÇÃO 1: Adicionar 'seed' e garantir 'shuffle=True' (padrão para treino, mas explícito)
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True, # Garantir o embaralhamento
    seed=RANDOM_SEED
)

# OTIMIZAÇÃO 2: Adicionar 'seed' no val_data e test_data
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=RANDOM_SEED
)

test_data = val_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    seed=RANDOM_SEED
)

num_classes = len(train_data.class_indices)
print(f"Número de classes: {num_classes}")
class_labels = list(train_data.class_indices.keys())

# -------------------------------
# IMPLEMENTAÇÃO DO TRANSFER LEARNING (VGG16)
# -------------------------------

# 1. Carregar o modelo base VGG16 pré-treinado no ImageNet
# - weights='imagenet': usa pesos pré-treinados
# - include_top=False: remove as camadas densas finais de classificação do ImageNet
# - input_shape: define o formato da imagem
base_model = VGG16(
    weights='imagenet', 
    include_top=False, 
    input_shape=(img_size[0], img_size[1], 3)
)

# 2. Congelar as camadas base
# Isso impede que os pesos pré-treinados sejam modificados durante as primeiras épocas,
# preservando o conhecimento aprendido.
for layer in base_model.layers:
    layer.trainable = False

# 3. Construir o novo classificador (Head)
model = Sequential([
    base_model, # Camadas convolucionais VGG16 (congeladas)
    Flatten(), # Achatando a saída da VGG16
    Dense(512, activation='relu'), # Nova camada densa maior
    Dropout(0.5),
    Dense(num_classes, activation='softmax') # Camada de saída para o nosso número de classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# CALLBACKS
# -------------------------------
model_path = os.path.join(base_dir, "plant_disease_model_vgg16_best.h5")
checkpoint = ModelCheckpoint(
    model_path, monitor='val_accuracy', save_best_only=True, verbose=1
)
earlystop = EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1
)

# -------------------------------
# TREINAMENTO
# -------------------------------
print("\n--- INICIANDO TREINAMENTO COM VGG16 (Transfer Learning) ---")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[checkpoint, earlystop]
)

# O restante do código (Avaliação e Plotagem) permanece o mesmo...

# -------------------------------
# AVALIAR NO TESTE
# -------------------------------
loss, accuracy = model.evaluate(test_data)
print(f"\nAcurácia no teste (VGG16 TL): {accuracy*100:.2f}%")

# -------------------------------
# PLOTAR GRÁFICOS
# -------------------------------
# ... (O código de plotagem é mantido)