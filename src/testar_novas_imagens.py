# testar_novas_imagens_unico.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -------------------------------
# CONFIGURAÇÃO DE CAMINHOS
# -------------------------------
model_path = r"C:\Users\julio\OneDrive\Área de Trabalho\tcc\plant_village_dataset\plant_disease_model_vgg16_best.h5"
images_dir = r"C:\Users\julio\OneDrive\Área de Trabalho\tcc\novas_imagens"

# -------------------------------
# CARREGAR MODELO
# -------------------------------
model = load_model(model_path)
print("✅ Modelo carregado com sucesso!")

# -------------------------------
# DEFINIR TAMANHO DE IMAGEM
# -------------------------------
img_size = (224, 224)  # mesmo tamanho usado no treino

# -------------------------------
# OBTER CLASSES DO TREINO
# -------------------------------
train_dir = r"C:\Users\julio\OneDrive\Área de Trabalho\tcc\plant_village_dataset\train"
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# -------------------------------
# FUNÇÃO DE PREDIÇÃO
# -------------------------------
def predict_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx] * 100
    print(f"{os.path.basename(image_path)} -> Classe prevista: {class_names[class_idx]} ({confidence:.2f}%)")

    # Mostrar imagem e gráfico
    plt.figure(figsize=(10,4))

    # Imagem
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Previsto: {class_names[class_idx]}")

    # Gráfico de barras
    plt.subplot(1,2,2)
    plt.barh(class_names, pred[0])
    plt.xlabel("Probabilidade")
    plt.xlim(0,1)
    plt.title("Distribuição de classes")
    plt.tight_layout()
    plt.show()

# -------------------------------
# TESTAR TODAS AS IMAGENS DA PASTA
# -------------------------------
for img_file in os.listdir(images_dir):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(images_dir, img_file)
        predict_image(img_path)
