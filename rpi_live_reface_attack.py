import cv2
import torch
import numpy as np
import torchvision.transforms as T
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
import requests  # Import the library to send HTTP requests

from defences import Randomization, BitDepthReduction, Jpeg_compression

RASPBERRY_PI_IP = "192.168.137.246"  # Replace with your Raspberry Pi's IP address

# U-Net based architecture for generating adversarial perturbations
class UNetAttackGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetAttackGenerator, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Output layer
        self.outconv = nn.Conv2d(64, out_channels, 1)
        self.tanh = nn.Tanh()  # To bound the perturbation

    def forward(self, x):
        # Encoding
        enc1 = self.enc1(x)
        pool1 = self.pool1(enc1)
        
        enc2 = self.enc2(pool1)
        pool2 = self.pool2(enc2)
        
        enc3 = self.enc3(pool2)
        pool3 = self.pool3(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)
        
        # Decoding
        upconv3 = self.upconv3(bottleneck)
        concat3 = torch.cat((upconv3, enc3), dim=1)
        dec3 = self.dec3(concat3)
        
        upconv2 = self.upconv2(dec3)
        concat2 = torch.cat((upconv2, enc2), dim=1)
        dec2 = self.dec2(concat2)
        
        upconv1 = self.upconv1(dec2)
        concat1 = torch.cat((upconv1, enc1), dim=1)
        dec1 = self.dec1(concat1)
        
        # Output
        perturbation = self.outconv(dec1)
        perturbation = self.tanh(perturbation) * 0.05  # Scale perturbation to be small
        
        # Return the perturbation, not the adversarial image
        return perturbation


def main():
    # Setup device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define class labels
    class_labels = ["Al", "Nhat", "Pierce", "Yaqi"]
    
    # Load the face recognition model
    face_model = InceptionResnetV1(
        classify=True,
        pretrained=None,
        num_classes=len(class_labels)
    ).to(device)
    
    face_model.load_state_dict(torch.load("facenet_4class.pth", map_location=device))
    face_model.eval()
    print("Face recognition model loaded successfully")
    
    # Collect all available attack models
    attack_models = {}
    for source in class_labels:
        for target in class_labels:
            if source == target:
                continue
                
            model_path = f"attack_{source}_to_{target}.pth"
            if os.path.exists(model_path):
                # Create and load attack model
                model = UNetAttackGenerator().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                
                attack_models[(source, target)] = model
                print(f"Loaded attack model: {source} → {target}")
    
    if len(attack_models) == 0:
        print("Error: No attack models found! Please train models first.")
        return
    
    print(f"Successfully loaded {len(attack_models)} attack models")
    
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
    
    # Define transformations
    transform = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # UI settings
    attack_active = False
    attack_strength = 1.0
    show_perturbation = False
    defence_active = False
    
    # Initialize target person for each possible detected person
    target_mapping = {}
    for person in class_labels:
        # Default target: choose next person in the list
        idx = (class_labels.index(person) + 1) % len(class_labels)
        target_mapping[person] = class_labels[idx]
    
    # Initialize selected face index (when multiple faces are detected)
    selected_face_idx = 0
    
    # Create window
    cv2.namedWindow("Multi-Directional Face Attack Demo", cv2.WINDOW_NORMAL)
    
    # Print controls
    print("\nControls:")
    print("  A: Toggle attack on/off")
    print("  P: Show/hide perturbation visualization")
    print("  +: Increase attack strength")
    print("  -: Decrease attack strength")
    print("  TAB: Switch between detected faces")
    print("  0-3: Change target person (0=Al, 1=Nhat, 2=Pierce, 3=Yaqi)")
    print("  Q: Quit")
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR->RGB for MTCNN detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = mtcnn.detect(rgb_frame)
        
        # Create display frame
        display_frame = frame.copy()
        
        # Draw status information at the top of the frame
        status_text = f"Attack: {'ON' if attack_active else 'OFF'} | Strength: {attack_strength:.1f}"
        cv2.putText(display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        defence_status_test = f"Defence: {'ON' if defence_active else 'OFF'}"
        cv2.putText(display_frame, defence_status_test, (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # If faces are detected
        if boxes is not None and len(boxes) > 0:
            # Limit selected face index to available faces
            num_faces = len(boxes)
            selected_face_idx = selected_face_idx % num_faces
            
            # Process each face
            for face_idx, (box, prob) in enumerate(zip(boxes, probs)):
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Highlight the selected face
                box_color = (0, 255, 255) if face_idx == selected_face_idx else (0, 255, 0)
                box_thickness = 3 if face_idx == selected_face_idx else 2
                
                # Check if box coordinates are valid
                if (x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and 
                    x2 < display_frame.shape[1] and y2 < display_frame.shape[0]):
                    
                    try:
                        # Crop and process the face
                        face_img = rgb_frame[y1:y2, x1:x2]
                        
                        # Check if the crop is valid
                        if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                            continue
                        
                        # Convert to PIL for transformations
                        face_img_pil = Image.fromarray(face_img)
                        face_tensor = transform(face_img_pil).unsqueeze(0).to(device)
                        
                        # Get original recognition result
                        with torch.no_grad():
                            original_output = face_model(face_tensor)
                            original_probs = F.softmax(original_output, dim=1)
                            original_pred = torch.argmax(original_probs, dim=1).item()
                            original_conf = original_probs[0][original_pred].item() * 100
                        
                        # Get recognized person
                        original_person = class_labels[original_pred]
                        
                        # Draw face rectangle
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, box_thickness)
                        
                        # Display face number
                        cv2.putText(display_frame, f"Face #{face_idx+1}", (x1, y1-40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                        
                        # Display original recognition
                        original_text = f"Detected: {original_person} ({original_conf:.1f}%)"
                        cv2.putText(display_frame, original_text, (x1, y1-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # If this is the selected face
                        if face_idx == selected_face_idx:
                            # Get target person for the detected identity
                            if original_person not in target_mapping:
                                # Initialize if not exist
                                idx = (class_labels.index(original_person) + 1) % len(class_labels)
                                target_mapping[original_person] = class_labels[idx]
                            
                            current_target = target_mapping[original_person]
                            target_idx = class_labels.index(current_target)
                            
                            # Display current attack target
                            target_text = f"Target: {current_target}"
                            cv2.putText(display_frame, target_text, (x1, y1+20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            
                            # Apply attack if activated
                            if attack_active:
                                # Check if we have the attack model for this transformation
                                attack_key = (original_person, current_target)
                                
                                if attack_key in attack_models:
                                    attack_model = attack_models[attack_key]
                                    
                                    # Generate perturbation
                                    with torch.no_grad():
                                        perturbation = attack_model(face_tensor) * attack_strength
                                        
                                        # Apply perturbation to create adversarial example
                                        adv_face_tensor = torch.clamp(face_tensor + perturbation, -1, 1)
                                        
                                        # Apply defence to face tensor
                                        if defence_active:
                                            # defence_1 = Randomization(device=device)
                                            # adv_face_tensor = defence_1(adv_face_tensor)

                                            defence_2 = Jpeg_compression(device=device)
                                            adv_face_tensor = defence_2(adv_face_tensor)

                                        # Get prediction on adversarial example
                                        adv_output = face_model(adv_face_tensor)
                                        adv_probs = F.softmax(adv_output, dim=1)
                                        adv_pred = torch.argmax(adv_probs, dim=1).item()
                                        adv_conf = adv_probs[0][adv_pred].item() * 100
                                    
                                    # Show perturbation if requested
                                    if show_perturbation:
                                        # Enhance perturbation for visualization (scale by 5)
                                        pert_display = (perturbation[0].permute(1, 2, 0).cpu().numpy() * 5) + 0.5
                                        pert_display = np.clip(pert_display * 255, 0, 255).astype(np.uint8)
                                        pert_display = cv2.cvtColor(pert_display, cv2.COLOR_RGB2BGR)
                                        pert_display = cv2.resize(pert_display, (x2-x1, y2-y1))
                                        
                                        # Add perturbation to display frame (right side)
                                        h, w = display_frame.shape[:2]
                                        margin = 20
                                        pert_x1 = w - pert_display.shape[1] - margin
                                        pert_y1 = margin
                                        pert_x2 = pert_x1 + pert_display.shape[1]
                                        pert_y2 = pert_y1 + pert_display.shape[0]
                                        
                                        display_frame[pert_y1:pert_y2, pert_x1:pert_x2] = pert_display
                                        cv2.putText(display_frame, "Perturbation (enhanced)", 
                                                   (pert_x1, pert_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   0.6, (255, 255, 255), 2)
                                    
                                    # Convert adversarial face back to image
                                    adv_face_np = ((adv_face_tensor[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
                                    adv_face_bgr = cv2.cvtColor(adv_face_np, cv2.COLOR_RGB2BGR)
                                    adv_face_resized = cv2.resize(adv_face_bgr, (x2-x1, y2-y1))
                                    
                                    # Display the adversarial face (right side of the frame)
                                    h, w = display_frame.shape[:2]
                                    adv_x1 = w - adv_face_resized.shape[1] - 20
                                    adv_y1 = 20 + (adv_face_resized.shape[0] + 40 if show_perturbation else 0)
                                    adv_x2 = adv_x1 + adv_face_resized.shape[1]
                                    adv_y2 = adv_y1 + adv_face_resized.shape[0]
                                    
                                    display_frame[adv_y1:adv_y2, adv_x1:adv_x2] = adv_face_resized
                                    
                                    # Add a text label for the adversarial face
                                    cv2.putText(display_frame, "Adversarial Face", 
                                               (adv_x1, adv_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                               0.6, (0, 0, 255), 2)
                                    
                                    # Display attack results
                                    attack_text = f"Attack result: {class_labels[adv_pred]} ({adv_conf:.1f}%)"
                                    attack_color = (0, 0, 255) if adv_pred == target_idx else (0, 165, 255)
                                    cv2.putText(display_frame, attack_text, (x1, y1+45),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, attack_color, 2)
                                    
                                    # Calculate attack success
                                    success = "SUCCESS" if adv_pred == target_idx else "FAILED"
                                    success_color = (0, 255, 0) if adv_pred == target_idx else (0, 0, 255)
                                    cv2.putText(display_frame, f"Attack: {success}", (x1, y1+70),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, success_color, 2)
                                    
                                    # Notify Raspberry Pi if the attack is successful
                                    if success == "SUCCESS":
                                        try:
                                            response = requests.post(f"http://{RASPBERRY_PI_IP}:5000/attack_success", json={
                                                "attacker": original_person,
                                                "target": current_target,
                                                "confidence": adv_conf
                                            })
                                            if response.status_code == 200:
                                                print(f"Successfully notified Raspberry Pi: {response.json()}")
                                            else:
                                                print(f"Failed to notify Raspberry Pi: {response.status_code}")
                                        except Exception as e:
                                            print(f"Error sending request to Raspberry Pi: {e}")
                                else:
                                    # No attack model available for this transformation
                                    cv2.putText(display_frame, f"No attack model for {original_person} → {current_target}", 
                                               (x1, y1+45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    except Exception as e:
                        print(f"Error processing face #{face_idx+1}: {e}")
        
        # Add instructions overlay
        h, w = display_frame.shape[:2]
        y_offset = h - 140
        
        # Draw semi-transparent background for instructions
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, y_offset), (350, y_offset + 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # Draw instructions
        cv2.putText(display_frame, "Controls:", (20, y_offset + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)   
        cv2.putText(display_frame, "A: Toggle attack", (20, y_offset + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "D: Toggle Defence", (20, y_offset + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "P: Show/hide perturbation", (20, y_offset + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "+/-: Adjust attack strength", (20, y_offset + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "TAB: Switch between faces", (20, y_offset + 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, "0-3: Change target", (20, y_offset + 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the frame
        cv2.imshow("Multi-Directional Face Attack Demo", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            attack_active = not attack_active
            print(f"Attack {'activated' if attack_active else 'deactivated'}")
        elif key == ord('d'):
            defence_active = not defence_active
            print(f"Defense {'activated' if defence_active else 'deactivated'}")
        elif key == ord('=') or key == ord('+'):
            attack_strength = min(attack_strength + 0.1, 3.0)
            print(f"Attack strength increased to {attack_strength:.1f}")
        elif key == ord('-'):
            attack_strength = max(attack_strength - 0.1, 0.1)
            print(f"Attack strength decreased to {attack_strength:.1f}")
        elif key == 9:  # TAB key
            if boxes is not None and len(boxes) > 0:
                selected_face_idx = (selected_face_idx + 1) % len(boxes)
                print(f"Selected face #{selected_face_idx+1}")
        elif key >= ord('0') and key <= ord('3'):
            # Change target for the selected face
            idx = key - ord('0')
            if idx < len(class_labels):
                if boxes is not None and len(boxes) > 0:
                    try:
                        # Get the currently selected face
                        box = boxes[selected_face_idx]
                        x1, y1, x2, y2 = [int(b) for b in box]
                        
                        # Get the recognized identity
                        face_img = rgb_frame[y1:y2, x1:x2]
                        face_img_pil = Image.fromarray(face_img)
                        face_tensor = transform(face_img_pil).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            original_output = face_model(face_tensor)
                            original_pred = torch.argmax(original_output, dim=1).item()
                            original_person = class_labels[original_pred]
                        
                        # Set new target
                        new_target = class_labels[idx]
                        if new_target != original_person:  # Can't target self
                            target_mapping[original_person] = new_target
                            print(f"Changed target for {original_person} to {new_target}")
                        else:
                            print(f"Cannot set target to self ({original_person})")
                    except Exception as e:
                        print(f"Error changing target: {e}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()