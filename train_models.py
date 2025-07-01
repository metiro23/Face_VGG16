import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json

class EmotionDataset(Dataset):
    """
    Custom Dataset class cho emotion recognition
    """
    
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        
        # Emotion labels
        self.emotion_labels = {
            'buon': 0,      # Buồn
            'vui': 1,       # Vui  
            'kho_chiu': 2,  # Khó chịu
            'tuc_gian': 3,  # Tức giận
            'bat_ngo': 4    # Bất ngờ
        }
        
        self.label_names = list(self.emotion_labels.keys())
        self.samples = []
        
        # Load tất cả samples
        self._load_samples()
    
    def _load_samples(self):
        """Load tất cả image paths và labels"""
        for emotion, label in self.emotion_labels.items():
            emotion_folder = os.path.join(self.data_folder, emotion)
            
            if os.path.exists(emotion_folder):
                for img_file in os.listdir(emotion_folder):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(emotion_folder, img_file)
                        self.samples.append((img_path, label))
        
        print(f"📊 Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class EmotionClassifier:
    """
    Main class cho emotion recognition training
    """
    
    def __init__(self, model_name='resnet34', num_classes=5):
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Class names
        self.class_names = ['buon', 'vui', 'kho_chiu', 'tuc_gian', 'bat_ngo']
        
        print(f"🔥 Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Training history
        self.train_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def _create_model(self):
        """Tạo model ResNet34 hoặc VGG16"""
        if self.model_name == 'resnet34':
            model = models.resnet34(pretrained=True)
            # Freeze early layers
            for param in model.parameters():
                param.requires_grad = False
            
            # Replace final layer
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            )
            
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            # Freeze early layers
            for param in model.features.parameters():
                param.requires_grad = False
            
            # Replace classifier
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
                nn.Linear(128, self.num_classes)
            )
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        print(f"✅ Created {self.model_name} model")
        return model
    
    def get_data_transforms(self):
        """Get data transforms cho training và validation"""
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def prepare_data(self, data_folder, batch_size=32, val_split=0.2):
        """Chuẩn bị data loaders"""
        train_transform, val_transform = self.get_data_transforms()
        
        # Load full dataset
        full_dataset = EmotionDataset(data_folder, transform=None)
        
        # Split dataset
        train_size = int((1 - val_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_indices, val_indices = random_split(
            range(len(full_dataset)), [train_size, val_size]
        )
        
        # Create datasets with appropriate transforms
        train_samples = [full_dataset.samples[i] for i in train_indices.indices]
        val_samples = [full_dataset.samples[i] for i in val_indices.indices]
        
        # Create datasets
        train_dataset = EmotionDataset.__new__(EmotionDataset)
        train_dataset.data_folder = data_folder
        train_dataset.transform = train_transform
        train_dataset.emotion_labels = full_dataset.emotion_labels
        train_dataset.label_names = full_dataset.label_names
        train_dataset.samples = train_samples
        
        val_dataset = EmotionDataset.__new__(EmotionDataset) 
        val_dataset.data_folder = data_folder
        val_dataset.transform = val_transform
        val_dataset.emotion_labels = full_dataset.emotion_labels
        val_dataset.label_names = full_dataset.label_names
        val_dataset.samples = val_samples
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=4)
        
        print(f"📊 Train samples: {len(train_samples)}")
        print(f"📊 Val samples: {len(val_samples)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train một epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        """Validate một epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, data_folder, num_epochs=50, batch_size=32, learning_rate=0.001):
        """Main training function"""
        print(f"🚀 Bắt đầu training {self.model_name}...")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(data_folder, batch_size)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        best_val_acc = 0.0
        best_model_path = f'models/best_{self.model_name}_model.pth'
        os.makedirs('models', exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'model_name': self.model_name
                }, best_model_path)
                print(f"💾 Saved best model with val_acc: {val_acc:.2f}%")
        
        print(f"\n🎉 Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save training history
        self.save_training_history()
        
        return best_val_acc
    
    def save_training_history(self):
        """Lưu training history"""
        history_path = f'models/{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f)
        print(f"📊 Training history saved to {history_path}")
    
    def plot_training_history(self):
        """Vẽ biểu đồ training history"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['train_loss'], label='Train Loss')
        plt.plot(self.train_history['val_loss'], label='Val Loss')
        plt.title(f'{self.model_name} - Training/Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_history['train_acc'], label='Train Acc')
        plt.plot(self.train_history['val_acc'], label='Val Acc')
        plt.title(f'{self.model_name} - Training/Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'models/{self.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function để train models
    """
    print("=" * 60)
    print("🎯 EMOTION RECOGNITION - MODEL TRAINING")
    print("=" * 60)
    
    # Đường dẫn data (frames đã extract từ data_preprocessing.py)
    data_folder = "data/frames"
    
    if not os.path.exists(data_folder):
        print(f"❌ Folder '{data_folder}' không tồn tại!")
        print("📝 Hãy chạy data_preprocessing.py trước")
        return
    
    # Training parameters
    num_epochs = 30
    batch_size = 16  # Giảm batch size nếu GPU memory không đủ
    learning_rate = 0.001
    
    # Train ResNet34
    print("\n🔴 Training ResNet34...")
    resnet_classifier = EmotionClassifier(model_name='resnet34')
    resnet_acc = resnet_classifier.train(
        data_folder=data_folder,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Plot history
    resnet_classifier.plot_training_history()
    
    # Train VGG16
    print("\n🔵 Training VGG16...")
    vgg_classifier = EmotionClassifier(model_name='vgg16')
    vgg_acc = vgg_classifier.train(
        data_folder=data_folder,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Plot history
    vgg_classifier.plot_training_history()
    
    # So sánh kết quả
    print("\n" + "="*60)
    print("📊 KẾT QUẢ TRAINING")
    print("="*60)
    print(f"ResNet34 Best Val Accuracy: {resnet_acc:.2f}%")
    print(f"VGG16 Best Val Accuracy: {vgg_acc:.2f}%")
    
    if resnet_acc > vgg_acc:
        print("🏆 ResNet34 performs better!")
    else:
        print("🏆 VGG16 performs better!")

if __name__ == "__main__":
    main()
