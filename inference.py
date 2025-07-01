import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class EmotionPredictor:
    """
    Class để predict emotion từ video hoặc webcam
    """
    
    def __init__(self, model_path, model_name='resnet34'):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class names
        self.class_names = ['buon', 'vui', 'kho_chiu', 'tuc_gian', 'bat_ngo']
        self.emotion_colors = {
            'buon': (255, 0, 0),      # Đỏ
            'vui': (0, 255, 0),       # Xanh lá
            'kho_chiu': (255, 165, 0), # Cam
            'tuc_gian': (255, 0, 255), # Tím
            'bat_ngo': (0, 255, 255)   # Xanh dương
        }
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Loaded {model_name} model from {model_path}")
    
    def _load_model(self, model_path):
        """Load trained model"""
        # Create model architecture
        if self.model_name == 'resnet34':
            model = models.resnet34(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 5)
            )
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 1024),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 5)
            )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_frame(self, frame):
        """
        Predict emotion từ một frame
        
        Args:
            frame: Frame từ video (numpy array, BGR format)
            
        Returns:
            tuple: (predicted_emotion, confidence_scores)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply transforms
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get prediction
            predicted_idx = torch.argmax(probabilities).item()
            predicted_emotion = self.class_names[predicted_idx]
            confidence = probabilities[predicted_idx].item()
            
            # Get all confidences
            confidences = {
                self.class_names[i]: probabilities[i].item() 
                for i in range(len(self.class_names))
            }
        
        return predicted_emotion, confidence, confidences
    
    def predict_video(self, video_path, output_path=None, show_video=True):
        """
        Predict emotion cho toàn bộ video
        
        Args:
            video_path: Đường dẫn video
            output_path: Đường dẫn lưu video output (optional)
            show_video: Hiển thị video realtime hay không
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Không thể mở video: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        emotion_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict emotion
            predicted_emotion, confidence, all_confidences = self.predict_frame(frame)
            emotion_history.append(predicted_emotion)
            
            # Draw prediction on frame
            frame_with_text = self._draw_prediction(frame, predicted_emotion, confidence, all_confidences)
            
            # Save frame
            if output_path:
                out.write(frame_with_text)
            
            # Show video
            if show_video:
                cv2.imshow('Emotion Recognition', frame_with_text)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Current emotion: {predicted_emotion}")
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Analyze emotion statistics
        self._analyze_emotion_stats(emotion_history)
        
        print(f"✅ Completed! Processed {frame_count} frames")
        if output_path:
            print(f"📹 Output video saved to: {output_path}")
    
    def _draw_prediction(self, frame, emotion, confidence, all_confidences):
        """Vẽ prediction lên frame"""
        frame_copy = frame.copy()
        
        # Draw main prediction
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        text = f"{emotion.upper()}: {confidence:.2f}"
        
        # Draw background rectangle
        cv2.rectangle(frame_copy, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(frame_copy, (10, 10), (400, 150), color, 2)
        
        # Draw main text
        cv2.putText(frame_copy, text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw all confidences
        y_offset = 70
        for i, (emo, conf) in enumerate(all_confidences.items()):
            conf_color = self.emotion_colors.get(emo, (255, 255, 255))
            conf_text = f"{emo}: {conf:.3f}"
            cv2.putText(frame_copy, conf_text, (20, y_offset + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, conf_color, 1)
        
        return frame_copy
    
    def _analyze_emotion_stats(self, emotion_history):
        """Phân tích thống kê emotion trong video"""
        if not emotion_history:
            return
        
        # Count emotions
        emotion_counts = {}
        for emotion in emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_frames = len(emotion_history)
        
        print("\n📊 EMOTION STATISTICS:")
        print("-" * 40)
        
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_frames) * 100
            print(f"{emotion.upper():>10}: {count:>5} frames ({percentage:.1f}%)")
        
        # Dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        print(f"\n🏆 Dominant emotion: {dominant_emotion[0].upper()} ({dominant_emotion[1]} frames)")
    
    def predict_webcam(self):
        """
        Predict emotion từ webcam realtime
        """
        print("🎥 Starting webcam emotion recognition...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Predict emotion
            predicted_emotion, confidence, all_confidences = self.predict_frame(frame)
            
            # Draw prediction
            frame_with_text = self._draw_prediction(frame, predicted_emotion, confidence, all_confidences)
            
            # Show frame
            cv2.imshow('Webcam Emotion Recognition', frame_with_text)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function để chạy inference
    """
    print("=" * 60)
    print("🎯 EMOTION RECOGNITION - INFERENCE")
    print("=" * 60)
    
    # Đường dẫn model (thay đổi theo model bạn muốn sử dụng)
    model_path = "models/best_resnet34_model.pth"  # hoặc best_vgg16_model.pth
    model_name = "resnet34"  # hoặc "vgg16"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file không tồn tại: {model_path}")
        print("📝 Hãy chạy train_models.py trước để train model")
        return
    
    # Khởi tạo predictor
    predictor = EmotionPredictor(model_path, model_name)
    
    # Menu
    while True:
        print("\n🎯 Chọn mode:")
        print("1. Predict từ video file")
        print("2. Predict từ webcam")
        print("3. Exit")
        
        choice = input("Nhập lựa chọn (1-3): ")
        
        if choice == '1':
            video_path = input("Nhập đường dẫn video: ")
            if os.path.exists(video_path):
                output_path = input("Nhập đường dẫn lưu video output (Enter để skip): ")
                if output_path == "":
                    output_path = None
                
                predictor.predict_video(video_path, output_path)
            else:
                print("❌ File video không tồn tại!")
        
        elif choice == '2':
            predictor.predict_webcam()
        
        elif choice == '3':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()
