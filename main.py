import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Base de dados (Nome, Ataque, Velocidade, Defesa, Nível de Poder)
robos = np.array([
    ["TitanX", 85, 70, 90, 82.5], ["ShadowCore", 75, 90, 60, 76.5], ["MechaRex", 90, 50, 85, 78.5],  
    ["NitroBot", 65, 95, 55, 70.5], ["IronClaw", 80, 60, 88, 78.4], ["NeonStriker", 88, 77, 69, 79.4],  
    ["SteelPhantom", 50, 80, 40, 58.0], ["Rusty", 40, 45, 35, 40.0], ["BlazeUnit", 95, 85, 80, 89.5],  
    ["OmegaKnight", 92, 70, 92, 85.8], ["CyberCrusher", 89, 65, 86, 81.0], ["TurboBot", 55, 99, 45, 64.5],  
    ["TitanZero", 78, 82, 74, 76.8], ["MechaFury", 85, 72, 79, 81.2], ["ElectroSting", 60, 85, 50, 66.0],  
    ["WarDroid", 95, 60, 90, 86.2],  
    ["ShadowGhost", 58, 87, 43, 63.5],  
    ["BladeCore", 80, 70, 85, 78.6],  
    ["VenomBot", 72, 68, 65, 69.5],  
    ["InfernoMech", 88, 90, 85, 86.0],  
    ["StealthUnit", 65, 92, 48, 69.8],  
    ["PlasmaKnight", 84, 73, 79, 78.9],  
    ["TitanBuster", 91, 65, 87, 82.0],  
    ["CyberSentinel", 76, 80, 72, 74.8],  
    ["RoboSpectre", 67, 77, 60, 69.2],  
    ["StormBreaker", 89, 81, 83, 83.5],  
    ["TeraCrusher", 90, 62, 91, 81.9],  
    ["SteelVortex", 50, 78, 40, 57.5],  
    ["NanoBot", 45, 40, 38, 39.0],  
    ["GigaTitan", 97, 88, 95, 92.0]  
])

# Título da página
st.set_page_config(page_title="🤖 SteelFight", layout="centered")

# Descrição do projeto
st.title("🤖 Bem-vindo a Arena SteelFight!")
st.write(
    """  
    🤖 Bem-vindo a Arena SteelFight!
    """
)

# Separando as variáveis
X = robos[:, 1:4].astype(float)  # Pegando Ataque, Velocidade e Defesa
Y = robos[:, 4].astype(float)    # Pegando Nível de Poder

# Criando e treinando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X, Y)

# Obtendo coeficientes
a1, a2, a3 = modelo.coef_
b = modelo.intercept_

# Criando as abas
aba1, aba2, aba3 = st.tabs(["Regras", "Oficina", "🔍 **Lista de robôs**"])

# Aba  Base de dados ultilizada
with aba3:
    df = pd.DataFrame(robos, columns=["Nome", "Ataque", "Velocidade", "Defesa", "Nível de Poder"])
    st.dataframe(df)


# Aba Modelo Supervisionado
with aba1:
    # Exibindo os resultados
    st.title("Modelo de Regressão Linear")
    st.write(
    """
        A Regressão Linear Múltipla é usada quando queremos prever um valor com base em múltiplas variáveis independentes. No nosso caso:

        **Variáveis Independentes (X):**
    """
    )
    st.write("🔹 x₁ = Ataque 💪")
    st.write("🔹 x₂ = Velocidade ⚡")
    st.write("🔹 x₃ = Defesa 🛡️")

    st.write(
    """
        **Variável Dependente (Y):**

        🔹 y = Nível de Poder
    """
    )
    st.write("A equação da regressão linear múltipla é:")
    st.latex(r"y = a_1 \cdot x_1 + a_2 \cdot x_2 + a_3 \cdot x_3 + b")
    st.write(
        """
        Onde:

        *a₁*, *a₂*, *a₃* são os coeficientes que determinam a influência de cada variável no resultado.     
        *b* é o intercepto, ou seja, o valor inicial quando todas as variáveis independentes são zero.
        """
    )

    st.write(
        """
        📊 **Análise dos Coeficientes da Regressão Linear**  

        Após treinar o modelo de regressão linear com nossa base de dados dos robôs, chegamos aos seguintes resultados:  
        """
    )
    st.write(f"🔹 **Coeficiente de Ataque (a1):** {a1:.4f}")
    st.write(f"🔹 **Coeficiente de Velocidade (a2):** {a2:.4f}")
    st.write(f"🔹 **Coeficiente de Defesa (a3):** {a3:.4f}")
    st.write(f"🔹 **Intercepto (b):** {b:.4f}")
    
    st.write("Cada coeficiente mostra **o impacto de cada atributo** (Ataque, Velocidade e Defesa) no nível de poder do robô:")
    st.write(f"🔹 Para cada **+1 ponto em Ataque**, o nível de poder aumenta **{a1:.2f} pontos**.")
    st.write(f"🔹 Para cada **+1 ponto em Velocidade**, o nível de poder aumenta **{a2:.2f} pontos**.")
    st.write(f"🔹 Para cada **+1 ponto em Defesa**, o nível de poder aumenta **{a3:.2f} pontos**.")
    st.write(f"🔹 O valor base do nível de poder (quando todos os atributos são 0) é **{b:.2f}**.")
    
    st.write("A equação final que representa nosso modelo é:")
    st.latex(rf"y = {a1:.4f} \cdot x_1 + {a2:.4f} \cdot x_2 + {a3:.4f} \cdot x_3 + {b:.4f}")


    ##### Criando um novo robô para prever nível de poder
    novo_robo = np.array([[80, 85, 75]])
    nivel_predito = modelo.predict(novo_robo)

    st.write(f"Se criarmos um novo robo com ***Ataque = 80***, ***Velocidade = 85*** e ***Defesa = 75*** o nível de poder previsto para esse robô é **{nivel_predito[0]:.2f}**.")

    # Valores do robô de exemplo
    ataque_exemplo = 80
    velocidade_exemplo = 85
    defesa_exemplo = 75

    # Cálculo manual
    nivel_calculado = (a1 * ataque_exemplo) + (a2 * velocidade_exemplo) + (a3 * defesa_exemplo) + b

    # Cálculos passo a passo
    st.latex(rf"""
    y = {a1:.2f} \cdot {ataque_exemplo} + {a2:.2f} \cdot {velocidade_exemplo} + {a3:.2f} \cdot {defesa_exemplo} + {b:.2f}
    """)

    st.latex(rf"""
    y = {a1 * ataque_exemplo:.2f} + {a2 * velocidade_exemplo:.2f} + {a3 * defesa_exemplo:.2f} + {b:.2f}
    """)

    st.latex(rf"""
    y = {nivel_calculado:.2f}
    """)


with aba2:
    df = pd.DataFrame(robos, columns=["Nome", "Ataque", "Velocidade", "Defesa", "Nível de Poder"])
    st.dataframe(df)
