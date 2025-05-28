import streamlit as st
import numpy as np
# import skfuzzy as fuzz
# import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
import pandas as pd

def calc_norm(M):
    if M.ndim == 1:
        sM = np.sum(M)
        return M/sM
    else:
        sM = np.sum(M, axis=0)
        return M / sM
    
alternatives = ["HP", "Lenovo", "Asus", "Acer", "Huawei"]
criteria = ["CPU", "GPU", "Memory", "Desain", "Harga"]

MPBk = np.array([
    [1/1, 2/1, 4/1, 3/1, 5/1],
    [1/2, 1/1, 3/1, 2/1, 4/1],
    [1/4, 1/3, 1/1, 1/2, 2/1],
    [1/3, 1/2, 2/1, 1/1, 3/1],
    [1/5, 1/4, 1/2, 1/3, 1/1]
])
w_MPB = calc_norm(MPBk)
m,n = w_MPB.shape
V = np.zeros(m)
for i in range(m):
    V[i] = np.sum(w_MPB[i, :])
w_MPB = V/m

AKB_CPU = np.array([
    [1/1, 2/1, 3/1, 3/1, 4/1],
    [1/2, 1/1, 2/1, 2/1, 3/1],
    [1/3, 1/2, 1/1, 2/1, 2/1],
    [1/3, 1/2, 1/2, 1/1, 2/1],
    [1/4, 1/3, 1/2, 1/2, 1/1]
])
w_CPU = calc_norm(AKB_CPU)
m, n = w_CPU.shape
V = np.zeros(m)
for i in range(m):
    V[i] = np.sum(w_CPU[i, :])
w_CPU = V/m

AKB_GPU = np.array([
    [1/1, 2/1, 3/1, 3/1, 4/1],
    [1/2, 1/1, 2/1, 2/1, 3/1],
    [1/3, 1/2, 1/1, 2/1, 2/1],
    [1/3, 1/2, 1/2, 1/1, 2/1],
    [1/4, 1/3, 1/2, 1/2, 1/1]
])
w_GPU = calc_norm(AKB_GPU)
m, n = w_GPU.shape
V = np.zeros(m)
for i in range(m):
    V[i] = np.sum(w_GPU[i, :])
w_GPU = V/m

ACM_Memory = np.array([16, 8, 8, 4, 8])
w_Memory = calc_norm(ACM_Memory)

AKB_Desain = np.array([
    [1/1, 2/1, 2/1, 3/1, 3/1],
    [1/2, 1/1, 1/1, 2/1, 2/1],
    [1/2, 1/1, 1/1, 2/1, 2/1],
    [1/3, 1/2, 1/2, 1/1, 1/1],
    [1/3, 1/2, 1/2, 1/1, 1/1]
])
w_Desain = calc_norm(AKB_Desain)
m, n = w_Desain.shape
V = np.zeros(m)
for i in range(m):
    V[i] = np.sum(w_Desain[i, :])
w_Desain = V / m

AKB_Harga = np.array([
    [1/1, 8/10, 9/10, 7/10, 8.5/10],
    [10/8, 1/1, 8/9, 7/8, 8/8.5],
    [10/9, 9/8, 1/1, 7/9, 8.5/9],
    [10/7, 8/7, 9/7, 1/1, 8.5/7],
    [10/8.5, 8.5/8, 9/8.5, 7/8.5, 1/1]
])
w_Harga = calc_norm(AKB_Harga)
m, n = AKB_Harga.shape
V = np.zeros(m)
for i in range(m):
    V[i] = np.sum(w_Harga[i, :])
w_Harga = V / m

st.write("Eigenvector Matriks Kriteria")
df_wMPB = pd.DataFrame(w_MPB, index=criteria, columns=['Eigenvector'])
st.dataframe(df_wMPB)

st.write("Eigenvector Alternatif")
wM = np.column_stack((w_CPU, w_GPU, w_Memory, w_Desain, w_Harga))
df_wM = pd.DataFrame(wM, index=alternatives, columns=criteria)
st.dataframe(df_wM)

st.write("Score Akhir")
MC_Scores = np.dot(wM, w_MPB)
df_MC_Scores = pd.DataFrame(MC_Scores, index=alternatives, columns=["Nilai Akhir"])
st.dataframe(df_MC_Scores)

Max_Score = np.max(MC_Scores)
Max_Score_Name = df_MC_Scores['Nilai Akhir'].idxmax()
st.write(f"Laptop terbaik adalah **{Max_Score_Name}** dengan nilai akhir **{Max_Score:.4f}**")

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(df_MC_Scores.index, df_MC_Scores["Nilai Akhir"], color="skyblue")

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')

ax.set_title("Skor Akhir Alternatif")
ax.set_ylabel("Nilai Akhir")
ax.set_ylim(0, df_MC_Scores["Nilai Akhir"].max() + 0.05)
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Tampilkan di Streamlit
st.pyplot(fig)