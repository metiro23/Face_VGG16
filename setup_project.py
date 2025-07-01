import os
import shutil

def create_project_structure():
    """
    Tạo cấu trúc folder cho emotion recognition project
    """
    print("📁 Tạo cấu trúc project...")
    
    # Main folders
    folders = [
        "data",
        "data/videos",
        "data/videos/buon",
        "data/videos/vui", 
        "data/videos/kho_chiu",
        "data/videos/tuc_gian",
        "data/videos/bat_ngo",
        "data/frames",
        "models",
        "results"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    # Tạo file .gitkeep để giữ empty folders
    gitkeep_folders = [
        "data/videos/buon",
        "data/videos/vui",
        "data/videos/kho_chiu", 
        "data/videos/tuc_gian",
        "data/videos/bat_ngo",
        "models",
        "results"
    ]
    
    for folder in gitkeep_folders:
        gitkeep_path = os.path.join(folder, ".gitkeep")
        with open(gitkeep_path, 'w') as f:
            f.write("")
    
    print("\n📋 Cấu trúc project đã được tạo:")
    print("emotion_recognition_project/")
    print("├── requirements.txt")
    print("├── data_preprocessing.py")
    print("├── train_models.py")
    print("├── inference.py")
    print("├── setup_project.py")
    print("├── README.md")
    print("├── data/")
    print("│   ├── videos/")
    print("│   │   ├── buon/        # Đặt video buồn vào đây")
    print("│   │   ├── vui/         # Đặt video vui vào đây")
    print("│   │   ├── kho_chiu/    # Đặt video khó chịu vào đây")
    print("│   │   ├── tuc_gian/    # Đặt video tức giận vào đây")
    print("│   │   └── bat_ngo/     # Đặt video bất ngờ vào đây")
    print("│   └── frames/          # Frames tự động tạo")
    print("├── models/              # Models tự động lưu")
    print("└── results/             # Kết quả training")
    
    print("\n📝 HƯỚNG DẪN TIẾP THEO:")
    print("1. Đặt video vào các folder tương ứng trong data/videos/")
    print("2. Chạy: python data_preprocessing.py")
    print("3. Chạy: python train_models.py")
    print("4. Chạy: python inference.py")

def check_requirements():
    """
    Kiểm tra và hướng dẫn cài đặt requirements
    """
    print("\n🔍 Kiểm tra requirements...")
    
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'numpy', 
        'matplotlib', 'scikit-learn', 'Pillow', 'tqdm', 
        'seaborn', 'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'Pillow':
                import PIL
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Cần cài đặt {len(missing_packages)} packages:")
        print("Chạy lệnh sau:")
        print("pip install -r requirements.txt")
        print("\nHoặc cài từng package:")
        for package in missing_packages:
            print(f"pip install {package}")
    else:
        print("\n✅ Tất cả packages đã được cài đặt!")

def main():
    """
    Main setup function
    """
    print("=" * 60)
    print("🎯 EMOTION RECOGNITION PROJECT SETUP")
    print("=" * 60)
    
    # Tạo cấu trúc folder
    create_project_structure()
    
    # Kiểm tra requirements
    check_requirements()
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT SETUP HOÀN THÀNH!")
    print("=" * 60)
    
    print("\n📋 BƯỚC TIẾP THEO:")
    print("1. Cài đặt packages: pip install -r requirements.txt")
    print("2. Đặt video data vào folders:")
    print("   - data/videos/buon/")
    print("   - data/videos/vui/") 
    print("   - data/videos/kho_chiu/")
    print("   - data/videos/tuc_gian/")
    print("   - data/videos/bat_ngo/")
    print("3. Chạy data preprocessing: python data_preprocessing.py")
    print("4. Chạy training: python train_models.py")
    print("5. Chạy inference: python inference.py")
    
    print("\n💡 TIPS:")
    print("- Mỗi emotion cần ít nhất 10-20 video")
    print("- Video nên dài 5-30 giây")
    print("- Chất lượng video tốt sẽ cho kết quả tốt hơn")
    print("- Sử dụng GPU để training nhanh hơn")

if __name__ == "__main__":
    main()
