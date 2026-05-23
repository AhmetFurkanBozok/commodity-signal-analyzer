import subprocess
import sys

# ---------------------------------------------------------
# Eksik Kütüphaneleri Otomatik Kontrol Etme ve Yükleme Bölümü
# ---------------------------------------------------------
required_libraries = ['yfinance', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn']
for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        print(f"[{lib}] kütüphanesi eksik, otomatik yükleniyor... Lütfen bekleyin.")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"[{lib}] başarıyla yüklendi.")
        except Exception as e:
            print(f"[{lib}] yüklenirken hata oluştu: {e}")

# Kütüphaneleri import etme
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Part 1: Veri İndirme ve Hazırlığı
# ---------------------------------------------------------
tickers = {
    'Oil': 'CL=F', 
    'Gold': 'GC=F', 
    'USD_Index': 'DX-Y.NYB', 
    'Copper': 'HG=F', 
    'S&P500': '^GSPC'
}

print("\nFinansal veriler Yahoo Finance üzerinden indiriliyor...")
df = yf.download(list(tickers.values()), start='2020-01-01', end='2024-01-01')['Close']
df.columns = [list(tickers.keys())[list(tickers.values()).index(col)] for col in df.columns]
df = df.dropna()

target = 'Oil'
features = [col for col in df.columns if col != target]

# Data Visualization (Grafik Çizimi)
print("\nGrafik oluşturuluyor... (Grafiği kapatmadan kodun devamı terminale yazdırılmaz!)")
plt.figure(figsize=(12, 6))
for col in df.columns:
    plt.plot(df.index, df[col] / df[col].iloc[0] * 100, label=col)

plt.title('Commodity Signals - Relative Price Changes (Base=100)')
plt.xlabel('Date')
plt.ylabel('Relative Price')
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------------
# Part 2: Normalization Table & The Correlation Test
# ---------------------------------------------------------
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

corr_matrix = df_normalized.corr()

print("\n" + "="*50)
print("--- Part 2: The Correlation Test ---")
print("="*50)
print("Correlation Matrix:\n", corr_matrix)

oil_correlations = corr_matrix[target].drop(target)
highest_corr_var = oil_correlations.abs().idxmax()
print(f"\nHighest correlation with Oil: {highest_corr_var} (Correlation Value: {oil_correlations[highest_corr_var]:.4f})")

# ---------------------------------------------------------
# Part 3: The ANN Architecture
# ---------------------------------------------------------
X = df_normalized[features]
y = df_normalized[target]

# Mimari Seçimi: 2 Gizli Katman (64 ve 32 nöron)
hidden_layer_sizes = (64, 32)
ann_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=2000, random_state=42, early_stopping=True)

print("\nYapay Sinir Ağı (ANN) eğitiliyor...")
ann_model.fit(X, y)

print("\n" + "="*50)
print("--- Part 3: The ANN Architecture ---")
print("="*50)
print(f"Design Choice: How many Hidden Layers? -> {len(hidden_layer_sizes)}")
print(f"How many Neurons per layer? -> {hidden_layer_sizes}")
print(f"Convergence: How many iterations did the model take to train? -> {ann_model.n_iter_}")

# ---------------------------------------------------------
# Part 4: Sensitivity & Signal Evaluation
# ---------------------------------------------------------
print("\n" + "="*50)
print("--- Part 4: Sensitivity & Signal Evaluation ---")
print("="*50)

base_inputs = X.mean().values.reshape(1, -1)
base_prediction = ann_model.predict(base_inputs)[0]

sensitivities = {}

for i, feature in enumerate(features):
    test_inputs = base_inputs.copy()
    test_inputs[0, i] = test_inputs[0, i] * 1.01 # Değişkeni %1 artırıyoruz
    
    new_prediction = ann_model.predict(test_inputs)[0]
    percent_change = ((new_prediction - base_prediction) / base_prediction) * 100
    sensitivities[feature] = percent_change
    print(f"Increase '{feature}' by 1%: Output (Oil) change = {percent_change:.4f}%")

# En düşük hassasiyete (etkiye) sahip değişken gürültüdür (Noise)
noise_variable = min(sensitivities, key=lambda k: abs(sensitivities[k]))
print(f"\nThe 'Noise' Filter: Based on your sensitivity analysis, '{noise_variable}' is actually 'Noise'.")
print("="*50)
