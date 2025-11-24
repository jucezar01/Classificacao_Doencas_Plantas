# 🌾 Classificação Automática de Doenças em Plantas com Redes Neurais Convolucionais

## 👤 Dados do Autor
**Autor:** Julio Cezar da Cunha  
**Instituição:** Fundação Hermínio Ometto (FHO)  
**Curso:** Engenharia de Computação  
**Data:** Novembro de 2025  

---

## 💡 Sobre o Projeto

Este projeto implementa e avalia modelos de **Redes Neurais Convolucionais (CNNs)** para a detecção automática de doenças em folhas de plantas. O foco está no uso de **Transfer Learning** com a arquitetura VGG16, permitindo maior desempenho em comparação a arquiteturas treinadas do zero.

O dataset utilizado é o **Plant Village**, dividido rigorosamente em:

- **70%** para Treino  
- **15%** para Validação  
- **15%** para Teste  

---

## 📊 Resultados e Comparação de Arquiteturas

| Abordagem | Arquitetura | Estratégia de Treinamento | Performance (Acurácia) | Observações |
| :--- | :--- | :--- | :--- | :--- |
| **Transfer Learning** | **VGG16** | Camadas base congeladas + Classificador personalizado | **[INSERIR ACURÁCIA FINAL AQUI]** | **Modelo de melhor desempenho**, adotado para teste final em campo. |
| **Treinamento do Zero** | CNN Customizada | 32 → 64 → 128 filtros | [INSERIR ACURÁCIA FINAL AQUI] | Performance inferior, usado como *baseline*. |

A arquitetura **VGG16** foi escolhida para os testes finais com **imagens reais de campo**, demonstrando excelente capacidade de generalização.

---

## 📁 Estrutura do Repositório

O repositório é organizado para a máxima reprodutibilidade:

| Arquivo/Pasta | Descrição |
| :--- | :--- |
| `src/` | Contém todos os scripts Python (`.py`). |
| `docs/` | Contém o documento acadêmico final em **LaTeX (IEEE)**. |
| `requirements.txt` | Lista exata de todas as bibliotecas e suas versões (ex: TensorFlow 2.20.0). |
| `LICENSE` | Licença de código aberto (MIT). |
| `.gitignore` | Configuração para ignorar arquivos grandes, como o modelo treinado (`*.h5`) e o dataset. |

### Principais Scripts (`src/`)

- **`plant_disease_complete_vgg16.py`** – Modelo VGG16 com Transfer Learning.  
- **`plant_disease_complete.py`** – CNN personalizada treinada do zero.  
- **`testar_novas_imagens.py`** – Teste em imagens externas reais.  
- **`organiza_plantvillage.py`** – Script para divisão de dados (70/15/15).

---

## ⚙️ Como Reproduzir o Projeto

### 1️⃣ Clonar o repositório e navegar até a pasta

git clone [https://github.com/jucezar01/Classificacao_Doencas_Plantas.git](https://github.com/jucezar01/Classificacao_Doencas_Plantas.git)
cd Classificacao_Doencas_Plantas

### 2️⃣ Criar e ativar o ambiente virtual

python -m venv .venv
.\.venv\Scripts\activate

### 3️⃣ Instalar dependências

pip install -r requirements.txt

### 4️⃣ Organizar o dataset
Coloque o dataset PlantVillage original na pasta raiz do projeto. O script irá organizar a estrutura (Treino/Validação/Teste).
python src/organiza_plantvillage.py

### 5️⃣ Treinar o modelo final
Para treinar o modelo de melhor desempenho:
python src/plant_disease_complete_vgg16.py

### 6️⃣ Testar com imagens externas
Para verificar a generalização em imagens reais de campo:
python src/testar_novas_imagens.py

📌 Observações Importantes
Os modelos treinados (.h5) e o dataset PlantVillage não são enviados ao GitHub devido ao limite de tamanho (conforme configurado no .gitignore).

O projeto é totalmente reprodutível utilizando apenas este repositório e o requirements.txt.
