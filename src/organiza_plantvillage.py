import os
import shutil
import random

# Caminho da pasta PlantVillage original
source_dir = r'C:\Users\julio\OneDrive\Área de Trabalho\plant_village\PlantVillage'

# Nova pasta organizada
base_dir = r'C:\Users\julio\OneDrive\Área de Trabalho\tcc\plant_village_dataset'
os.makedirs(base_dir, exist_ok=True)

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)

# Percentual de cada conjunto
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Para cada classe
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Lista apenas arquivos de imagem válidos
    files = []
    for f in os.listdir(class_path):
        full_path = os.path.join(class_path, f)
        if os.path.isfile(full_path) and f.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(f)

    if not files:
        print(f"⚠️ Nenhuma imagem encontrada na classe: {class_name}")
        continue

    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Criar pastas da classe nos splits
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)

    # Copiar arquivos
    for i, file in enumerate(files):
        src_file = os.path.join(class_path, file)
        if i < n_train:
            dst_file = os.path.join(base_dir, 'train', class_name, file)
        elif i < n_train + n_val:
            dst_file = os.path.join(base_dir, 'val', class_name, file)
        else:
            dst_file = os.path.join(base_dir, 'test', class_name, file)
        shutil.copy2(src_file, dst_file)

print("✅ Organização concluída sem erros!")
