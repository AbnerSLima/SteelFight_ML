import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Título da página
st.set_page_config(page_title="🤖 SteelFight", layout="centered")

# Descrição do projeto
st.title("🤖 Arena SteelFight!")
st.write(
    """  
    🤖 Bem-vindo a Arena SteelFight!
    """
)