import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import json
import base64
import io
import httpx
from siamese_net import SiameseNetwork

# ---------------------------------------------------------
# Image Preprocessing (for Siamese Network)
# ---------------------------------------------------------
def preprocess_image(img):
    target_size = (224, 224)
    if img.mode != 'L':
        img = img.convert('L')
        
    original_width, original_height = img.size
    target_width, target_height = target_size
    
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_img = Image.new('L', target_size, 255)
    
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    new_img.paste(img, (offset_x, offset_y))
    
    img_array = np.array(new_img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    return img_tensor

# ---------------------------------------------------------
# Load Siamese Model Once
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = SiameseNetwork()

model_path = "signature_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
else:
    print(f"Warning: {model_path} not found.")

# ---------------------------------------------------------
# Helper: Convert PIL Image to Base64 for Ollama API
# ---------------------------------------------------------
def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ---------------------------------------------------------
# LLaVA Vision LLM Analysis via Ollama
# ---------------------------------------------------------
def llm_analyze_signatures(anchor_image, test_image):
    try:
        # Combine both images side by side into one image for better context
        w1, h1 = anchor_image.size
        w2, h2 = test_image.size
        max_h = max(h1, h2)
        combined = Image.new('RGB', (w1 + w2 + 20, max_h), (255, 255, 255))
        combined.paste(anchor_image, (0, 0))
        combined.paste(test_image, (w1 + 20, 0))
        
        img_b64 = pil_to_base64(combined)
        
        prompt = (
            "Analyze these two handwritten signatures side by side for a handwriting comparison study. "
            "The LEFT signature is the reference. The RIGHT signature is the test sample. "
            "Perform a detailed visual comparison of the two images. Specifically describe: "
            "1. The slant, angle, and baseline consistency. "
            "2. The formation of specific loops, curves, and character terminals. "
            "3. The connecting strokes between letters. "
            "4. Any unique flourishes or idiosyncratic marks. "
            "List your specific observations regarding similarities and differences. "
            "Conclude whether the visual characteristics of the test sample are consistent with the reference signature."
        )
        
        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            },
            timeout=120.0
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response from LLM")
        else:
            return f"Error from Ollama: HTTP {response.status_code}"
            
    except httpx.ConnectError:
        return "⚠️ Ollama is not running. Please start it with: brew services start ollama"
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------------------------------------
# Combined Prediction Function
# ---------------------------------------------------------
def verify_signature(anchor_image, test_image, threshold, use_llm):
    if anchor_image is None or test_image is None:
        return "Please upload both images!", "", ""
    
    # 1. Siamese Network Prediction
    img1 = preprocess_image(anchor_image).to(device)
    img2 = preprocess_image(test_image).to(device)
    
    with torch.no_grad():
        output1, output2 = model(img1, img2)
        distance = F.pairwise_distance(output1, output2).item()
        
    if distance < threshold:
        result_text = "✅ GENUINE"
        details = f"Distance: {distance:.4f} (Threshold: {threshold:.2f})"
    else:
        result_text = "❌ FORGED"
        details = f"Distance: {distance:.4f} (Threshold: {threshold:.2f})"
    
    # 2. LLM Analysis (optional)
    llm_result = ""
    if use_llm:
        llm_result = llm_analyze_signatures(anchor_image, test_image)
    else:
        llm_result = "Toggle 'Enable LLM Analysis' to get AI-powered explanation."
        
    return result_text, details, llm_result

# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------
with gr.Blocks(title="Signature Verification AI") as demo:
    gr.Markdown("# ✍️ Signature Verification AI")
    gr.Markdown("Upload a known genuine signature on the left and the signature to test on the right. The AI uses a **Siamese Neural Network** for mathematical comparison and optionally a **Vision LLM (LLaVA)** for a written forensic analysis.")
    
    with gr.Row():
        anchor_input = gr.Image(type="pil", label="1. Known Genuine Signature (Anchor)")
        test_input = gr.Image(type="pil", label="2. Signature to Test")
    
    with gr.Row():
        threshold_slider = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.1, label="Strictness Threshold (Higher = More Forgiving)")
        use_llm_toggle = gr.Checkbox(label="🧠 Enable LLM Analysis (requires Ollama + LLaVA)", value=False)

    verify_btn = gr.Button("🔍 Verify Signature", variant="primary", size="lg")
    
    with gr.Row():
        result_output = gr.Textbox(label="Prediction Result", lines=1)
        details_output = gr.Textbox(label="Mathematical Details", lines=1)
    
    llm_output = gr.Textbox(label="🧠 LLM Forensic Analysis", lines=8)
        
    verify_btn.click(
        fn=verify_signature, 
        inputs=[anchor_input, test_input, threshold_slider, use_llm_toggle], 
        outputs=[result_output, details_output, llm_output]
    )

if __name__ == "__main__":
    print("Launching Web Interface...")
    demo.launch(server_name="127.0.0.1", server_port=7860)
