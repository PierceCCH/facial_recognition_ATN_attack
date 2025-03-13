import cv2
import torch
import numpy as np
import torchvision.transforms as T
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Suppose 4-person classification: Al, Nhat, Pierce, Yaqi
###############################################################################
class_labels = ["Al", "Nhat", "Pierce", "Yaqi"]
num_classes = 4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) MTCNN: for face detection
    #    keep_all=False => only the first bounding box is returned if multiple
    #    margin=0 => no extra border, image_size=160 => for InceptionResnetV1
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    
    # 2) InceptionResnetV1 for 4-class face classification
    #    Must match your training in "train_victim_facenet.py"
    model = InceptionResnetV1(
        classify=True,
        pretrained=None,
        num_classes=num_classes
    ).to(device)
    
    # load your trained weights: "facenet_4class.pth"
    model.load_state_dict(torch.load("facenet_4class.pth", map_location=device))
    model.eval()
    
    # 3) open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Define transformation for the face image to match your model's expected input
    # Using the same transforms as in your training code
    transform = T.Compose([
        T.Resize((160, 160)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGR->RGB for MTCNN detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 4) detect faces
        #    boxes shape can be (n,4) if multiple faces or (4,) if single
        boxes, probs = mtcnn.detect(rgb_frame)
        if boxes is not None:
            # unify shape => if single face => shape(4,)
            if len(boxes.shape) == 1:
                boxes = np.expand_dims(boxes, axis=0)  # now (1,4)
            
            for (x1, y1, x2, y2) in boxes:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                try:
                    # Directly crop the face from the RGB frame
                    face_img = rgb_frame[y1:y2, x1:x2]
                    
                    # Check if the crop is valid
                    if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                        continue
                    
                    # Convert the cropped face to a PIL Image for transformation
                    face_img_pil = T.ToPILImage()(np.array(face_img))
                    
                    # Apply transforms to get a normalized tensor
                    face_tensor = transform(face_img_pil).unsqueeze(0).to(device)
                    
                    # Forward pass through model
                    with torch.no_grad():
                        outputs = model(face_tensor)  # => [1,4]
                        
                        # Apply softmax to get probabilities
                        probs = F.softmax(outputs, dim=1)
                        
                        # Get highest probability and index
                        confidence, pred_idx = torch.max(probs, dim=1)
                        pred_label = class_labels[pred_idx.item()]
                        confidence_value = confidence.item() * 100  # Convert to percentage
                    
                    # Display prediction with confidence
                    label_text = f"{pred_label}: {confidence_value:.1f}%"
                    
                    # Choose color based on confidence (green for high, yellow for medium, red for low)
                    if confidence_value > 80:
                        color = (0, 255, 0)  # Green
                    elif confidence_value > 50:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 0, 255)  # Red
                    
                    # Draw label with confidence
                    cv2.putText(frame, label_text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # Draw all class probabilities
                    y_offset = y2 + 20
                    for i, class_name in enumerate(class_labels):
                        prob_text = f"{class_name}: {probs[0][i].item()*100:.1f}%"
                        cv2.putText(frame, prob_text, (x1, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        y_offset += 20
                
                except Exception as e:
                    print(f"Error processing face: {e}")
        
        cv2.imshow("Facenet 4-class Real-time", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()