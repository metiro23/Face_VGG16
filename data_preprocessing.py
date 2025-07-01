import cv2
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

class VideoFrameExtractor:
    """
    Class để extract frames từ video cho training emotion recognition
    """
    
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        self.emotion_labels = {
            'buon': 0,      # Buồn
            'vui': 1,       # Vui
            'kho_chiu': 2,  # Khó chịu
            'tuc_gian': 3,  # Tức giận
            'bat_ngo': 4    # Bất ngờ
        }
    
    def extract_frames_from_video(self, video_path, max_frames=30):
        """
        Extract frames từ một video
        
        Args:
            video_path: Đường dẫn video
            max_frames: Số frame tối đa extract
        
        Returns:
            list: Danh sách frames đã resize
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Lỗi: Không thể mở video {video_path}")
            return []
        
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Tính step để lấy frames đều nhau
        if total_frames <= max_frames:
            step = 1
        else:
            step = total_frames // max_frames
        
        frame_indices = range(0, total_frames, step)[:max_frames]
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame
                frame = cv2.resize(frame, self.output_size)
                frames.append(frame)
        
        cap.release()
        return frames
    
    def process_dataset(self, data_folder, output_folder, max_frames_per_video=30):
        """
        Xử lý toàn bộ dataset video thành frames để training
        
        Args:
            data_folder: Folder chứa video theo cấu trúc:
                        data_folder/
                        ├── buon/
                        │   ├── video1.mp4
                        │   └── video2.mp4
                        ├── vui/
                        │   ├── video1.mp4
                        │   └── video2.mp4
                        └── ...
            output_folder: Folder lưu frames đã extract
            max_frames_per_video: Số frame tối đa extract từ mỗi video
        """
        print("🎬 Bắt đầu xử lý dataset video...")
        
        # Tạo folder output
        os.makedirs(output_folder, exist_ok=True)
        
        total_frames = 0
        
        for emotion in self.emotion_labels.keys():
            emotion_folder = os.path.join(data_folder, emotion)
            output_emotion_folder = os.path.join(output_folder, emotion)
            
            if not os.path.exists(emotion_folder):
                print(f"⚠️  Không tìm thấy folder: {emotion_folder}")
                continue
            
            os.makedirs(output_emotion_folder, exist_ok=True)
            
            # Lấy danh sách video files
            video_files = [f for f in os.listdir(emotion_folder) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            print(f"📁 Xử lý emotion '{emotion}': {len(video_files)} videos")
            
            frame_count = 0
            
            for video_file in tqdm(video_files, desc=f"Processing {emotion}"):
                video_path = os.path.join(emotion_folder, video_file)
                
                # Extract frames
                frames = self.extract_frames_from_video(video_path, max_frames_per_video)
                
                # Lưu frames
                video_name = os.path.splitext(video_file)[0]
                
                for i, frame in enumerate(frames):
                    frame_filename = f"{video_name}_frame_{i+1:03d}.jpg"
                    frame_path = os.path.join(output_emotion_folder, frame_filename)
                    
                    # Convert numpy array to PIL Image and save
                    pil_image = Image.fromarray(frame)
                    pil_image.save(frame_path, quality=95)
                    frame_count += 1
            
            print(f"   ✅ Đã extract {frame_count} frames cho emotion '{emotion}'")
            total_frames += frame_count
        
        print(f"\n🎉 Hoàn thành! Tổng cộng {total_frames} frames đã được extract")
        print(f"📁 Frames được lưu tại: {output_folder}")
        
        # Hiển thị thống kê
        self.show_dataset_stats(output_folder)
    
    def show_dataset_stats(self, dataset_folder):
        """
        Hiển thị thống kê dataset
        """
        print("\n📊 THỐNG KÊ DATASET:")
        print("-" * 40)
        
        total_images = 0
        
        for emotion in self.emotion_labels.keys():
            emotion_folder = os.path.join(dataset_folder, emotion)
            
            if os.path.exists(emotion_folder):
                image_files = [f for f in os.listdir(emotion_folder) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                count = len(image_files)
                total_images += count
                
                print(f"{emotion.upper():>10}: {count:>6} images")
        
        print("-" * 40)
        print(f"{'TOTAL':>10}: {total_images:>6} images")
        
        if total_images > 0:
            print(f"\nPhân bố:")
            for emotion in self.emotion_labels.keys():
                emotion_folder = os.path.join(dataset_folder, emotion)
                if os.path.exists(emotion_folder):
                    image_files = [f for f in os.listdir(emotion_folder) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    percentage = (len(image_files) / total_images) * 100
                    print(f"  {emotion}: {percentage:.1f}%")

def main():
    """
    Main function để chạy data preprocessing
    """
    print("=" * 60)
    print("🎯 EMOTION RECOGNITION - DATA PREPROCESSING")
    print("=" * 60)
    
    # Khởi tạo extractor
    extractor = VideoFrameExtractor(output_size=(224, 224))
    
    # Đường dẫn folders
    # Thay đổi đường dẫn này theo cấu trúc folder của bạn
    data_folder = "data/videos"  # Folder chứa video gốc
    output_folder = "data/frames"  # Folder lưu frames
    
    print(f"📁 Input folder: {data_folder}")
    print(f"📁 Output folder: {output_folder}")
    print("\n📝 Cấu trúc folder cần có:")
    print("data/videos/")
    print("├── buon/")
    print("│   ├── video1.mp4")
    print("│   └── video2.mp4")
    print("├── vui/")
    print("│   ├── video1.mp4") 
    print("│   └── video2.mp4")
    print("├── kho_chiu/")
    print("├── tuc_gian/")
    print("└── bat_ngo/")
    
    # Kiểm tra folder tồn tại
    if not os.path.exists(data_folder):
        print(f"\n❌ Folder '{data_folder}' không tồn tại!")
        print("📝 Hãy tạo folder và đặt video theo cấu trúc trên")
        return
    
    # Bắt đầu xử lý
    extractor.process_dataset(
        data_folder=data_folder,
        output_folder=output_folder,
        max_frames_per_video=30  # Extract tối đa 30 frames/video
    )
    
    print("\n✅ Data preprocessing hoàn thành!")
    print("🚀 Bây giờ bạn có thể chạy training script!")

if __name__ == "__main__":
    main()
