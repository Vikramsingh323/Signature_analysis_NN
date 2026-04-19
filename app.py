import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import base64
import io
import httpx
from siamese_net import SiameseNetwork

# ---------------------------------------------------------
# GAN Generator Definition (Needed to load signature_generator.pth)
# ---------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

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
# Load Models
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
siamese_model = SiameseNetwork()
siamese_model_path = "signature_model.pth"
if os.path.exists(siamese_model_path):
    siamese_model.load_state_dict(torch.load(siamese_model_path, map_location=device, weights_only=True))
    siamese_model.to(device)
    siamese_model.eval()

# Load GAN
gan_latent_dim = 100
generator = Generator(gan_latent_dim)
gan_model_path = "signature_generator.pth"
if os.path.exists(gan_model_path):
    generator.load_state_dict(torch.load(gan_model_path, map_location=device, weights_only=True))
    generator.to(device)
    generator.eval()

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_synthetic_forgery():
    if not os.path.exists(gan_model_path):
        return None
    noise = torch.randn(1, gan_latent_dim, 1, 1, device=device)
    with torch.no_grad():
        fake_tensor = generator(noise).detach().cpu().squeeze()
    
    # Post-process: Tanh (-1 to 1) -> 0 to 255
    fake_array = ((fake_tensor.numpy() + 1) / 2 * 255).astype(np.uint8)
    fake_img = Image.fromarray(fake_array, mode='L')
    # Resize to 224x224 for UI
    fake_img = fake_img.resize((224, 224), Image.Resampling.LANCZOS)
    return fake_img

def llm_analyze_signatures(anchor_images, test_image):
    try:
        anchor_image = Image.open(anchor_images[0])
        w1, h1 = anchor_image.size
        w2, h2 = test_image.size
        max_h = max(h1, h2)
        combined = Image.new('RGB', (w1 + w2 + 20, max_h), (255, 255, 255))
        combined.paste(anchor_image, (0, 0))
        combined.paste(test_image, (w1 + 20, 0))
        img_b64 = pil_to_base64(combined)
        prompt = ("Analyze these two handwritten signatures side by side for a handwriting comparison study. "
                  "The LEFT signature is reference. The RIGHT signature is the sample. "
                  "Perform detailed visual comparison of slant, loops, and connecting strokes.")
        response = httpx.post("http://localhost:11434/api/generate",
                              json={"model": "llava", "prompt": prompt, "images": [img_b64], "stream": False},
                              timeout=120.0)
        return response.json().get("response", "No response from LLM")
    except Exception as e:
        return f"LLM Analysis Error: {str(e)}"

def verify_signature(anchor_files, test_image, threshold, use_llm):
    if not anchor_files or test_image is None:
        return "Please upload at least one anchor and one test image!", "", ""
    img_test = preprocess_image(test_image).to(device)
    distances = []
    for anchor_file in anchor_files:
        anchor_img_pil = Image.open(anchor_file.name)
        img_anchor = preprocess_image(anchor_img_pil).to(device)
        with torch.no_grad():
            out_anchor, out_test = siamese_model(img_anchor, img_test)
            dist = F.pairwise_distance(out_anchor, out_test).item()
            distances.append(dist)
    avg_distance = sum(distances) / len(distances)
    if avg_distance < threshold:
        result_text = "✅ GENUINE"
        details = f"Average Distance: {avg_distance:.4f} (Against {len(distances)} anchors)"
    else:
        result_text = "❌ FORGED"
        details = f"Average Distance: {avg_distance:.4f} (Threshold: {threshold:.2f})"
    llm_result = llm_analyze_signatures([f.name for f in anchor_files], test_image) if use_llm else "Toggle 'Enable LLM Analysis' for AI explanation."
    return result_text, details, llm_result

# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------
with gr.Blocks(title="Signature AI: Ultimate Edition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✍️ Signature AI: Ultimate Edition")
    gr.Markdown("1. Upload a **Gallery of Genuine Signatures**. 2. Upload a test signature OR **Generate a Synthetic Forgery** to challenge the AI.")
    
    with gr.Row():
        with gr.Column():
            anchor_input = gr.File(file_count="multiple", file_types=["image"], label="1. Gallery of Genuine Signatures")
        with gr.Column():
            test_input = gr.Image(type="pil", label="2. Signature to Test")
            gen_fake_btn = gr.Button("🤖 Generate AI Synthetic Forgery", variant="secondary")
    
    with gr.Row():
        threshold_slider = gr.Slider(minimum=0.1, maximum=3.0, value=1.0, step=0.1, label="Strictness Threshold")
        use_llm_toggle = gr.Checkbox(label="🧠 Enable LLM Analysis", value=False)

    verify_btn = gr.Button("🔍 Verify against Gallery", variant="primary", size="lg")
    
    with gr.Row():
        result_output = gr.Textbox(label="Final Verdict")
        details_output = gr.Textbox(label="Mathematical Analytics")
    
    llm_output = gr.Textbox(label="🧠 LLM Forensic Analysis", lines=8)
    
    # Event Handlers
    gen_fake_btn.click(fn=generate_synthetic_forgery, outputs=test_input)
    verify_btn.click(fn=verify_signature, inputs=[anchor_input, test_input, threshold_slider, use_llm_toggle], outputs=[result_output, details_output, llm_output])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
