# Time-Series-Forecasting-Models 📈

This repository contains Python scripts for time-series extrapolation using **Long Short-Term Memory (LSTM)** and a **Transformer** model. Each model is implemented in a separate script for focused experimentation.

The goal is to train a model on an initial portion of a signal and then use it to forecast or "extrapolate" the signal's far future behavior.



## ✨ Features

* **Separate Implementations**: Focused scripts for the LSTM (`lstm_forecasting.py`) and the Transformer (`transformer_forecasting.py`).
* **PyTorch Models**: Both architectures are built from scratch using PyTorch.
* **Automated Preprocessing**: Scripts handle loading data, scaling it to the `[-1, 1]` range, and calculating an appropriate downsampling factor.
* **End-to-End Workflow**: Each script covers the full pipeline from data loading and training to extrapolation and visualization.
* **Clear Visualizations**: Generates plots showing the original signal, the model's predictions, and the start of the extrapolation.

---

## 📂 Project Structure

```
.
├── lstm_forecasting.py         # Script for the LSTM model
├── transformer_forecasting.py  # Script for the Transformer model
├── requirements.txt            # Required Python packages
├── README.md                   # 
└── data/
    └── o11_model1.txt          # Example signal data
```

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/rodrigorezende1/Time-Series-Forecasting-Models]([https://github.com/rodrigorezende1/RNN-Based-Time-Series-Extrapolation](https://github.com/rodrigorezende1/Time-Series-Forecasting-Models))
cd Time-Series-Forecasting-Models
```

### 2. Set Up a Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure the File Path

Before running, you **must** update the path to your data file inside the script you want to run. Open either `lstm_forecasting.py` or `transformer_forecasting.py` and modify this line:

```python
# Change this to the location of your signal file
FILE_PATH = "path/to/your/data/o11_model1.txt"
```

### 5. Run the Model

You can now run either the LSTM or the Transformer script.

```bash
# To run the LSTM model
python lstm_forecasting.py

# To run the Transformer model
python transformer_forecasting.py
```
The script will start the training process and display a plot with the results upon completion.

---

## 🧠 Model Details

### LSTM
The **Long Short-Term Memory** network is a type of Recurrent Neural Network (RNN) that excels at learning from sequential data. It uses a system of gates to decide which information to store in its memory and which to forget, allowing it to capture long-term dependencies in the time series.

### Transformer
The **Transformer** model, originally from the field of natural language processing, relies on a **self-attention mechanism**. Instead of processing data sequentially, it can weigh the importance of all points in the input sequence simultaneously. This allows it to capture complex patterns and relationships across the time series, making it a powerful alternative to RNNs.
