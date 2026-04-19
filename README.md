# Signature Verification AI ✍️

A professional-grade Signature Verification system powered by **Siamese Neural Networks** and **Vision LLMs**. This tool compares two handwritten signatures to determine authenticity using both deep learning mathematics and forensic AI analysis.

## 🚀 Features

- **Siamese Network:** A custom-built PyTorch CNN that calculates the mathematical distance between two signature pen-stroke patterns.
- **AI Forensic Analysis:** Integrates **Ollama + LLaVA** to provide natural-language explanations of signature similarities and differences.
- **Interactive Web UI:** A beautiful Gradio interface supporting drag-and-drop image uploads.
- **Adjustable Strictness:** Includes a dynamic threshold slider to fine-tune the model's sensitivity.
- **100% Local:** Everything runs on your machine—no data is sent to the cloud.

## 🛠️ Technology Stack

- **Core:** Python, PyTorch
- **Computer Vision:** Pillow, NumPy, TorchVision
- **Web Interface:** Gradio
- **AI Explanations:** Ollama (LLaVA Vision Model)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vikramsingh323/Signature_analysis_NN.git
   cd Signature_analysis_NN
   ```

2. **Set up Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install torch torchvision Pillow numpy gradio httpx
   ```

4. **Install Ollama (Optional, for AI Explanations):**
   - Download from [ollama.com](https://ollama.com)
   - Run: `ollama pull llava`

## 🕹️ How to Use

### 1. Launch the Web Interface
Start the interactive UI where you can drag and drop signatures for instant verification:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`.

### 2. Run Batch Testing
Test the model against 20 random samples from the dataset and generate an accuracy report:
```bash
python test_signature.py
```

### 3. Training
If you want to retrain the model on new data:
```bash
python train_model.py
```

## 🧠 Model Architecture

The system uses a **Siamese Network** with a 3-layer Convolutional Neural Network (CNN) backbone. It converts signature images into 128-dimensional embedding vectors and uses **Contrastive Loss** to pull genuine signatures together and push forged signatures apart.

---
Created with ❤️ by [Vikram Singh](https://github.com/Vikramsingh323)
