# 🤖 SteelFight ML - Arena de Batalha de Robôs

Bem-vindo ao **SteelFight ML**, um projeto de aprendizado de máquina onde você pode **montar seu próprio robô de batalha** e descobrir em qual liga ele pertence! Utilizamos **Regressão Linear Múltipla** para estimar o nível de poder do robô e **K-Means** para classificá-lo em categorias de batalha. 🚀⚙️

## 🔥 Funcionalidades
✅ **Monte seu robô** escolhendo peças para Corpo, Braços e Pernas.  
✅ **Predição do nível de poder** com aprendizado de máquina.  
✅ **Classificação automática** em ligas de batalha com K-Means.  
✅ **Visualização 3D** dos robôs no espaço de atributos (Ataque, Velocidade e Defesa).  
✅ **Base de dados com 30 robôs** para treinamento do modelo.

## 🛠️ Tecnologias Utilizadas
- Python 🐍
- Streamlit 🎨 (Interface interativa)
- NumPy 🔢 (Manipulação de arrays)
- Pandas 🗃️ (Manipulação de dados)
- Matplotlib 📊 (Visualização de dados)
- Scikit-Learn 🤖 (Modelos de Machine Learning)

## 🚀 Como Executar o Projeto

### 1️⃣ Clone o repositório
```bash
  git clone https://github.com/AbnerSLima/SteelFight-ML.git
  cd SteelFight-ML
```

### 2️⃣ Crie um ambiente virtual (Opcional, mas recomendado)
```bash
  python -m venv venv
  source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3️⃣ Instale as dependências
```bash
  pip install -r requirements.txt
```

### 4️⃣ Execute o projeto
```bash
  streamlit run main.py
```

## 🏆 Ligas de Batalha
Os robôs são classificados automaticamente em **três ligas** de acordo com seus atributos:

🏴 **Liga Sucata** → Robôs mais fracos  
🔵 **Liga Titânio** → Robôs balanceados  
🟠 **Liga Overdrive** → Os mais poderosos  

## 📜 Modelos de Machine Learning
### 🔹 Regressão Linear Múltipla
Utilizada para prever o **nível de poder** dos robôs a partir de três variáveis:
- **x₁** = Ataque 💪
- **x₂** = Velocidade ⚡
- **x₃** = Defesa 🛡️

A equação usada é:
```math
y = a_1 * x_1 + a_2 * x_2 + a_3 * x_3 + b
```

### 🔹 K-Means Clustering
Usado para agrupar robôs em diferentes categorias com base em seus atributos.
O algoritmo calcula a **distância euclidiana** entre cada robô e os centroides dos clusters.

```math
d = \sqrt{(x_1 - C_1)^2 + (x_2 - C_2)^2 + (x_3 - C_3)^2}
```

## 🎨 Visualizações
✅ **Gráficos interativos** para exibição das classificações.  
✅ **Representação 3D** dos clusters no espaço Ataque x Velocidade x Defesa.  

---

💡 **Desenvolvido por [Abner Silva](https://github.com/AbnerSLima)** 🚀

