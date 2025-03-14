import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression


# Base de dados (Nome, Ataque, Velocidade, Defesa, Nível de Poder)
robos = np.array([
    ["TitanX", 85, 70, 90, 82.5], ["ShadowCore", 75, 90, 60, 76.5], ["MechaRex", 90, 50, 85, 78.5],  
    ["NitroBot", 65, 95, 55, 70.5], ["IronClaw", 80, 60, 88, 78.4], ["NeonStriker", 88, 77, 69, 79.4],  
    ["SteelPhantom", 50, 80, 40, 58.0], ["Rusty", 40, 45, 35, 40.0], ["BlazeUnit", 95, 85, 80, 89.5],  
    ["OmegaKnight", 92, 70, 92, 85.8], ["CyberCrusher", 89, 65, 86, 81.0], ["TurboBot", 55, 99, 45, 64.5],  
    ["TitanZero", 78, 82, 74, 76.8], ["MechaFury", 85, 72, 79, 81.2], ["ElectroSting", 60, 85, 50, 66.0],  
    ["WarDroid", 95, 60, 90, 86.2], ["ShadowGhost", 58, 87, 43, 63.5], ["BladeCore", 80, 70, 85, 78.6],  
    ["VenomBot", 72, 68, 65, 69.5], ["InfernoMech", 88, 90, 85, 86.0], ["StealthUnit", 65, 92, 48, 69.8],  
    ["PlasmaKnight", 84, 73, 79, 78.9], ["TitanBuster", 91, 65, 87, 82.0], ["CyberSentinel", 76, 80, 72, 74.8],  
    ["RoboSpectre", 67, 77, 60, 69.2], ["StormBreaker", 89, 81, 83, 83.5], ["TeraCrusher", 90, 62, 91, 81.9],  
    ["SteelVortex", 50, 78, 40, 57.5], ["NanoBot", 45, 40, 38, 39.0], ["GigaTitan", 97, 88, 95, 92.0]  
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

# Aba Base de dados ultilizada
with aba3:
    df = pd.DataFrame(robos, columns=["Nome", "Ataque", "Velocidade", "Defesa", "Nível de Poder"])
    st.dataframe(df)

# Aba Treino
with aba1:
    # Modelo Supervisionado
    st.title("🔍 Modelo Supervisionado - Regressão Linear Múltipla")
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


    # Criando um novo robô para prever nível de poder
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
    st.title("🧮 Modelo Não Supervisionado - K-Means")

    # 📌 Explicação K-Means
    st.write(
        """       
        O **K-Means** é um algoritmo de aprendizado não supervisionado que agrupa robôs automaticamente com base em suas características (Ataque, Velocidade e Defesa).     
        
        1️⃣ **Definição das Variáveis**       
        No nosso caso, temos três variáveis para classificar os robôs:

            x₁ = Ataque 💪      
            x₂ = Velocidade ⚡      
            x₃ = Defesa 🛡️      

        Cada robô é representado como um ponto no espaço (x₁, x₂, x₃).  

        2️⃣ **Escolhendo o Número de Grupos (Clusters)**      
        Definimos que queremos 3 grupos de robôs:

            ⚪ Iniciantes → Robôs mais fracos
            🟡 Intermediários → Robôs balanceados
            🔴 Avançados → Os mais poderosos

        Então, configuramos o K-Means com k = 3 clusters.

        3️⃣ **Inicialização Aleatória dos Centroides**        
        O K-Means começa escolhendo aleatoriamente 3 pontos iniciais (centroides) para representar os grupos.

        Digamos que os centroides iniciais escolhidos sejam:
        """
    )
    st.latex(r"C_1 = (50, 70, 45), \quad C_2 = (80, 85, 75), \quad C_3 = (95, 90, 85)")
    st.write(
        """ 
        Os centroides são pontos fictícios que representam o centro de cada grupo.

  
        4️⃣ **Cada robô é atribuído ao grupo mais próximo**.      
        Agora, para cada robô, calculamos a distância euclidiana até cada centroide.        

        A distância euclidiana entre um robô (x₁, x₂, x₃) e um centroide (C₁, C₂, C₃) é dada por:
        """
    )
    st.latex(r"d = \sqrt{(x_1 - C_1)^2 + (x_2 - C_2)^2 + (x_3 - C_3)^2}")
    st.write("Exemplo para o robô 'TitanX' (85, 70, 90):")
    st.write("🔹 Distância até C₁ = (50, 70, 45):")
    st.latex(r"d_1 = \sqrt{(85 - 50)^2 + (70 - 70)^2 + (90 - 45)^2}")
    st.latex(r"d_1 = \sqrt{(35)^2 + 0 + (45)^2} = \sqrt{1225 + 2025} = \sqrt3250 \approx 57.0")
    st.write("🔹 Distância até C₂ = (80, 85, 75):")
    st.latex(r"d_2 = \sqrt{(85 - 80)^2 + (70 - 85)^2 + (90 - 75)^2}")
    st.latex(r"d_2 = \sqrt{(5)^2 + (-15)^2 + (15)^2} = \sqrt{25 + 225 + 225} = \sqrt{475} \approx 21.8")
    st.write("🔹 Distância até C₃ = (95, 90, 85):")
    st.latex(r"d_3 = \sqrt{(85 - 95)^2 + (70 - 90)^2 + (90 - 85)^2}")
    st.latex(r"d_3 = \sqrt{(-10)^2 + (-20)^2 + (5)^2} = \sqrt{100 + 400 + 25} = \sqrt{525} \approx 22.9")
    st.write(
        """
        Como **d₂** (21.8) é a menor distância, **TitanX** será atribuído ao cluster **C₂ (Intermediário)**.

        Esse processo é repetido para **todos os robôs**.  
        """
    )
    st.write(
        """
        5️⃣ **Recalculando os Centroides**       
        Depois de classificar todos os robôs, o K-Means recalcula a **posição média** dos robôs dentro de cada grupo para encontrar um novo centroide.      
        
        Se um grupo contém robôs: 
        """
    )
    st.latex(r"(80,85,75),(78,82,74),(84,73,79)")
    st.write("O novo centroide será a média de cada atributo:")
    st.latex(r"C_{\text{novo}} = \left( \frac{80 + 78 + 84}{3}, \frac{85 + 82 + 73}{3}, \frac{75 + 74 + 79}{3} \right)")
    st.latex(r"C_{\text{novo}} = \left( \frac{242}{3}, \frac{240}{3}, \frac{228}{3} \right)")
    st.latex(r"C_{\text{novo}} = (80.6, 80, 76)")
    st.write("Esse processo é repetido **até que os centroides não mudem mais.**")
    st.write("6️⃣ **Resultado Final**")
    st.write(
        """
        Após algumas iterações, o modelo agrupa os robôs em três categorias com base no nível de poder:     
            🔹 ⚪ Iniciantes → Robôs mais fracos       
            🔹 🟡 Intermediários → Robôs balanceados       
            🔹 🔴 Avançados → Os mais poderosos        
        Agora, sempre que adicionarmos um **novo robô**, o K-Means determinará automaticamente **qual categoria ele pertence**!
        """
    )

    # Aplicando K-Means
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    # Obtendo os rótulos (clusters)
    clusters = kmeans.labels_

    # Criando DataFrame atualizado com categorias
    df_clusters = pd.DataFrame(robos, columns=["Nome", "Ataque", "Velocidade", "Defesa", "Nível de Poder"])
    df_clusters["Categoria"] = clusters

    # Ordenando clusters corretamente (do mais fraco ao mais forte)
    clusters_ordenados = sorted(range(num_clusters), key=lambda i: kmeans.cluster_centers_[i, 0])
    categorias = {
        clusters_ordenados[0]: "⚪ Iniciante",
        clusters_ordenados[1]: "🟡 Intermediário",
        clusters_ordenados[2]: "🔴 Avançado"
    }

    # Aplicando a classificação correta
    df_clusters["Categoria"] = df_clusters["Categoria"].map(categorias)

    # 📌 Exibindo explicação dos cálculos
    st.write(
        """
        📊 **Cálculo do K-Means - Passo a Passo**  
        
        O modelo inicia com **3 centroides aleatórios** e ajusta os grupos iterativamente até encontrar a melhor classificação.  
        """
    )

    # Fórmula da Distância Euclidiana
    st.write("🎯 **1️⃣ Fórmula usada para calcular a distância entre um robô e um centroide:**")
    st.latex(r"d = \sqrt{(x_1 - C_1)^2 + (x_2 - C_2)^2 + (x_3 - C_3)^2}")

    st.write(
        """
        🔹 *x₁*, *x₂* e *x₃* representam os valores de **Ataque**, **Velocidade** e **Defesa** de um robô.        
        🔹 *C₁*, *C₂* e *C₃* representam as coordenadas do **centroide do cluster** no mesmo espaço de atributos.
        """
    )

    st.write(
        """
        ---
        🎯 **2️⃣ Exemplo prático com o robô TitanX:**  

        O robô **TitanX** tem os seguintes atributos:  
        - **Ataque** = 85  
        - **Velocidade** = 70  
        - **Defesa** = 90  

        Suponha que um dos centroides iniciais seja **(50, 70, 45)**.  
        Aplicamos a fórmula:
        """
    )

    # Aplicação da fórmula com valores do TitanX
    st.latex(r"d = \sqrt{(85 - 50)^2 + (70 - 70)^2 + (90 - 45)^2}")

    st.write("🎯 **3️⃣ Resolvendo os cálculos:**")

    # Etapas do cálculo
    st.latex(r"d = \sqrt{(35)^2 + (0)^2 + (45)^2}")
    st.latex(r"d = \sqrt{1225 + 0 + 2025}")
    st.latex(r"d = \sqrt{3250} \approx 57.0")

    st.write(
        """
        ---
        📌 **O modelo faz esse cálculo para todos os robôs e ajusta os grupos até encontrar a melhor classificação!**  
        """
    )

    st.write("📊 **Visualização dos Grupos - Gráfico 3D**")

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Plotando os pontos no gráfico 3D
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=clusters, cmap="viridis", s=100)

    # Definindo os rótulos dos eixos
    ax.set_xlabel("Ataque")
    ax.set_ylabel("Velocidade")
    ax.set_zlabel("Defesa")
    ax.set_title("Classificação dos Robôs - K-Means (Gráfico 3D)")

    # Exibir gráfico no Streamlit
    st.pyplot(fig)

    # Exibindo a tabela com categorias
    st.write("📜 **Tabela de Classificação dos Robôs**")
    st.dataframe(df_clusters)

# Aba Oficina
with aba2:
    st.title("🔍 Modelo Supervisionado - Regressão Linear Múltipla")