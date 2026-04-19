import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import random
from siamese_net import SiameseNetwork

def preprocess_image(img_path):
    target_size = (224, 224)
    img = Image.open(img_path)
    
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

def predict_signature_distance(model, device, anchor_path, test_path):
    img1 = preprocess_image(anchor_path).to(device)
    img2 = preprocess_image(test_path).to(device)
    
    with torch.no_grad():
        output1, output2 = model(img1, img2)
        distance = F.pairwise_distance(output1, output2).item()
        
    return distance

def extract_owner(filename):
    base_name = filename.split('_')[0].replace('.png', '').replace('.jpg', '')
    if len(base_name) >= 8:
        return base_name[-3:]
    return "Unknown"

def run_batch_test(model_path, base_dir, num_tests=10, threshold=1.0):
    print("=========================================")
    print("      BATCH SIGNATURE VERIFICATION       ")
    print("=========================================\n")
    print(f"Loading Model from '{model_path}'...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Collect files
    real_dir = os.path.join(base_dir, 'real')
    forge_dir = os.path.join(base_dir, 'forge')
    
    real_files = [f for f in os.listdir(real_dir) if f.endswith('.png')]
    forge_files = [f for f in os.listdir(forge_dir) if f.endswith('.png')]

    # Map owner to their real and forged files
    real_dict = {}
    forge_dict = {}
    
    for f in real_files:
        owner = extract_owner(f)
        if owner not in real_dict:
            real_dict[owner] = []
        real_dict[owner].append(os.path.join(real_dir, f))
        
    for f in forge_files:
        owner = extract_owner(f)
        if owner not in forge_dict:
            forge_dict[owner] = []
        forge_dict[owner].append(os.path.join(forge_dir, f))

    correct_predictions = 0
    total_tests = num_tests * 2 # Genuine + Forged
    
    print("\n--- Testing Genuine Pairs ---")
    for i in range(num_tests):
        # Pick a random owner that has at least 2 real signatures
        valid_owners = [o for o, files in real_dict.items() if len(files) >= 2]
        owner = random.choice(valid_owners)
        
        anchor = random.choice(real_dict[owner])
        test_img = random.choice(real_dict[owner])
        while test_img == anchor:
            test_img = random.choice(real_dict[owner])
            
        dist = predict_signature_distance(model, device, anchor, test_img)
        pred = "GENUINE" if dist < threshold else "FORGED"
        success = "✅" if pred == "GENUINE" else "❌"
        if pred == "GENUINE":
            correct_predictions += 1
            
        print(f"Test {i+1}: Dist {dist:.4f} -> {pred} {success}")

    print("\n--- Testing Forged Pairs ---")
    for i in range(num_tests):
        # Pick a random owner that has both real and forged signatures
        valid_owners = [o for o in real_dict.keys() if o in forge_dict and len(forge_dict[o]) > 0]
        owner = random.choice(valid_owners)
        
        anchor = random.choice(real_dict[owner])
        test_img = random.choice(forge_dict[owner])
            
        dist = predict_signature_distance(model, device, anchor, test_img)
        pred = "GENUINE" if dist < threshold else "FORGED"
        success = "✅" if pred == "FORGED" else "❌"
        if pred == "FORGED":
            correct_predictions += 1
            
        print(f"Test {i+1}: Dist {dist:.4f} -> {pred} {success}")

    accuracy = (correct_predictions / total_tests) * 100
    print("\n=========================================")
    print(f"FINAL ACCURACY: {accuracy:.1f}% ({correct_predictions}/{total_tests})")
    print("=========================================")

if __name__ == "__main__":
    run_batch_test("signature_model.pth", "Processed_Dataset/dataset1", num_tests=10, threshold=1.0)
