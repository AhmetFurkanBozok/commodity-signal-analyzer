# The Commodity Signal Analyzer

This project is a data-driven engineering analytical tool designed to extract meaningful signals and filter noise from volatile financial markets. Using historical market data, the system evaluates the relationships between various independent macroeconomic factors and their direct influence on Crude Oil prices through Statistical Correlation and Artificial Neural Networks (ANN).

## 🚀 Key Features
* **Automated Data Pipeline:** Fetches real-time financial data directly from Yahoo Finance (`yfinance`).
* **Feature Normalization:** Implements `MinMaxScaler` to guarantee unbiased neural network weight distribution.
* **Predictive Modeling:** Multi-Layer Perceptron (MLP) Regressor topology to simulate non-linear market behaviors.
* **Sensitivity Analysis:** Determines the exact elasticity of output trends given a structural $+1\%$ shift in independent parameters.

---

## 📊 Evaluation & Assignment Results

### Part 1: Data Visualization
* **Observation:** When plotting the relative price actions, a strong co-movement ("Signal") is observed between specific energy and metal commodities, whereas broader market indexes represent decentralized fluctuations ("Noise").

### Part 2: Normalization & The Correlation Test
The features were scaled into a $[0, 1]$ boundaries. 

* **Highest Correlation Asset:** `[Buraya kodun çıktı verdiği en yüksek ilişkili varlığı yaz, örn: Copper]`
* **Correlation Value:** `[Buraya çıkan korelasyon değerini yaz, örn: 0.7452]`

### Part 3: The ANN Architecture
* **Design Choice:** 2 Hidden Layers
* **Layer Topology:** 64 Neurons (Layer 1) $\rightarrow$ 32 Neurons (Layer 2)
* **Activation Function:** ReLU
* **Convergence Iterations:** Took `[Buraya terminaldeki n_iter_ değerini yaz]` iterations to close and optimize weights.

### Part 4: Sensitivity & Signal Evaluation
By inducing a strategic $+1\%$ impulse to each asset independent matrix, the structural variance on Oil prices yielded:
* **Gold (+1%):** `[...]%` output change.
* **USD Index (+1%):** `[...]%` output change.
* **Copper (+1%):** `[...]%` output change.
* **S&P 500 (+1%):** `[...]%` output change.

* **The "Noise" Filter:** Based on structural sensitivity, `[En düşük etkiyi veren değişkenin adı]` is flagged as **Noise** due to its minimal impact on the model's objective function.

---

## 🧠 Part 5: Engineering Conclusion
**Would you trust your model? Why or why not?**

*Engineering Perspective:* The model provides a reliable framework for understanding historical macroeconomic links. However, from a strict systems control perspective, deploying this model for automated high-frequency execution without integrating structural dynamic memory layers (such as LSTM or recurrent nodes) poses financial risks. While the current ANN successfully mapping static non-linear dependencies, financial markets are subject to stochastic regime shifts and black swan anomalies that standard MLPs cannot dynamically inherit. Therefore, I trust this model as an **analytical signal evaluator**, but not as an autonomous trading mechanism.

---

## 🛠️ Prerequisites & Local Execution

1. Clone the workspace repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/commodity-signal-analyzer.git](https://github.com/YOUR_USERNAME/commodity-signal-analyzer.git)
