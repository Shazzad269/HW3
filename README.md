# HW3
# Stock Price Forecasting Using Deep Learning 
 
This project focuses on forecasting the future price of Microsoft (MSFT) stock using different deep learning techniques on historical stock data from multiple companies.  
It explores multiple model architectures, model compression with knowledge distillation, and advanced state-space modeling with Mamba-style layers.
 
 
Project Structure
 
1.1 Data Preparation
- StockDataset class is created to handle sequential data for stock prices.
- Yahoo Finance API is used to download closing price data for 20 feature stocks and MSFT as the target.
- All features and target data are normalized (standardized).
- Dataset is split into training (70%), validation (15%), and testing (15%) sets.
- DataLoaders are prepared using PyTorch Lightning framework.

1(a) RNN with LSTM Layer
- A deep learning model is built using:
  - LSTM (Long Short-Term Memory) layers to capture long-term dependencies in stock time series.
  - Fully connected (Linear) layer at the end to predict the next day's MSFT stock price.
- Key features:
  - Two stacked LSTM layers with dropout for regularization.
  - Final dense layer outputs a single scalar price prediction.
- The model is trained and evaluated.
-Training/Validation/Test MSE are reported, and loss curves are plotted.

1(b) RNN with Self-Attention Layer
- Another model is designed that:
  - First applies a 1D Convolution to capture local temporal patterns.
  - Then passes through a Multi-Head Self-Attention Layer to focus on important time steps.
  - Finally uses Dense layers to predict MSFT stock price.
- The model is trained and evaluated similarly to the LSTM model.
- Training/Validation/Test MSE are plotted, and total parameters are counted.
 
1(c) Knowledge Distillation: Shrinking the Model
- A smaller Student model is created by:
  - Reducing the embedding dimension and number of attention heads by half.
- Knowledge Distillation is applied:
  - The student learns both from the true MSFT labels and from the teacher model’s outputs.
- The Student model achieves comparable performance with roughly half the parameter count.
 
1(d) Mamba-Based Modeling (with Adaptation)
- Initially attempted to use the Mamba-SSM layer.
- Note:
  Mamba-SSM could not be installed because it requires CUDA/C++ compilation, which is not supported on Google Colab.
- As a solution, a FakeMambaLayer was implemented:
  - Mimics Mamba's structure using two Linear layers, GELU activation, and a skip connection.
- A complete model was built using this FakeMamba layer, trained, and evaluated.

How to Run
 
1. Install the necessary Python libraries:
 
   pip install torch pytorch-lightning yfinance matplotlib
