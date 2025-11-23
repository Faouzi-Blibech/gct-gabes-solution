"""
Phosphogypsum Quality Inspection - CV Model Training
=====================================================
This script trains a MobileNetV3-based model for:
1. Whiteness Score Regression (0-100%)
2. Purity Grade Classification (A, B, C, D)
3. Defect Detection (Binary)

For hackathon: Uses synthetic data generation if real dataset unavailable.
"""

import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

# ============================================
# 1. SYNTHETIC DATA GENERATOR (For Hackathon)
# ============================================

class SyntheticPGDataset:
    """
    Generates synthetic phosphogypsum images for training.
    In production, replace with real camera captures.
    """
    
    @staticmethod
    def generate_pg_image(grade='A', size=(224, 224)):
        """Generate synthetic PG sample image based on grade."""
        
        # Base colors for different grades (RGB)
        grade_colors = {
            'A': (245, 243, 238),  # Very white/cream
            'B': (232, 228, 217),  # Light cream
            'C': (212, 207, 192),  # Darker cream
            'D': (184, 176, 160),  # Grayish/contaminated
        }
        
        base_color = np.array(grade_colors[grade], dtype=np.float32)
        
        # Create base image with slight variations
        img = np.zeros((size[0], size[1], 3), dtype=np.float32)
        for i in range(3):
            noise = np.random.normal(0, 5, size)
            img[:, :, i] = np.clip(base_color[i] + noise, 0, 255)
        
        # Add texture (granular appearance)
        texture = np.random.normal(0, 3, size)
        for i in range(3):
            img[:, :, i] = np.clip(img[:, :, i] + texture, 0, 255)
        
        # Convert to uint8 BEFORE drawing circles/lines (cv2 requirement)
        img = img.astype(np.uint8)
        
        # Add impurities based on grade
        num_impurities = {'A': 0, 'B': 3, 'C': 8, 'D': 15}[grade]
        for _ in range(num_impurities):
            cx, cy = np.random.randint(20, size[0]-20), np.random.randint(20, size[1]-20)
            radius = np.random.randint(3, 12)
            color_offset = np.random.randint(-80, -30)
            # Calculate impurity color (darker spots)
            impurity_color = (
                max(0, int(base_color[0]) + color_offset),
                max(0, int(base_color[1]) + color_offset),
                max(0, int(base_color[2]) + color_offset)
            )
            cv2.circle(img, (cx, cy), radius, impurity_color, -1)
        
        # Add defects for lower grades
        has_defect = grade in ['C', 'D'] and np.random.random() > 0.5
        if has_defect:
            # Add crack-like defect
            pt1 = (np.random.randint(30, size[0]-30), np.random.randint(30, size[1]-30))
            pt2 = (pt1[0] + np.random.randint(-40, 40), pt1[1] + np.random.randint(-40, 40))
            cv2.line(img, pt1, pt2, (100, 95, 85), thickness=2)
        
        # Calculate whiteness score (0-100)
        whiteness = np.mean(img) / 255 * 100
        whiteness = min(100, max(0, whiteness + np.random.normal(0, 2)))
        
        return img, whiteness, has_defect, grade
    
    @staticmethod
    def generate_dataset(num_samples=1000, save_dir='./pg_dataset'):
        """Generate full synthetic dataset."""
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/images', exist_ok=True)
        
        metadata = []
        grades = ['A', 'B', 'C', 'D']
        samples_per_grade = num_samples // 4
        
        print(f"Generating {num_samples} synthetic PG images...")
        
        for grade in grades:
            for i in tqdm(range(samples_per_grade), desc=f"Grade {grade}"):
                img, whiteness, has_defect, _ = SyntheticPGDataset.generate_pg_image(grade)
                
                filename = f"{grade}_{i:04d}.jpg"
                filepath = f"{save_dir}/images/{filename}"
                cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                metadata.append({
                    'filename': filename,
                    'grade': grade,
                    'grade_idx': grades.index(grade),
                    'whiteness': round(whiteness, 2),
                    'has_defect': int(has_defect)
                })
        
        # Save metadata
        with open(f'{save_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {save_dir}")
        return metadata


# ============================================
# 2. PYTORCH DATASET CLASS
# ============================================

class PGDataset(Dataset):
    """PyTorch Dataset for Phosphogypsum images."""
    
    def __init__(self, metadata, img_dir, transform=None):
        self.metadata = metadata
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, item['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Labels
        whiteness = torch.tensor(item['whiteness'], dtype=torch.float32)
        grade = torch.tensor(item['grade_idx'], dtype=torch.long)
        has_defect = torch.tensor(item['has_defect'], dtype=torch.float32)
        
        return image, whiteness, grade, has_defect


# ============================================
# 3. MULTI-HEAD MODEL ARCHITECTURE
# ============================================

class PGQualityModel(nn.Module):
    """
    Multi-output model for PG quality inspection.
    
    Architecture:
    - Backbone: MobileNetV3-Small (pretrained)
    - Head 1: Whiteness Regression (1 output)
    - Head 2: Grade Classification (4 classes)
    - Head 3: Defect Detection (binary)
    """
    
    def __init__(self, num_grades=4, pretrained=True):
        super(PGQualityModel, self).__init__()
        
        # Load pretrained MobileNetV3-Small
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Get the number of features from backbone
        num_features = self.backbone.classifier[0].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Head 1: Whiteness Regression
        self.whiteness_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output 0-1, multiply by 100 for percentage
        )
        
        # Head 2: Grade Classification
        self.grade_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_grades)
        )
        
        # Head 3: Defect Detection
        self.defect_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        shared = self.shared_fc(features)
        
        # Multi-head outputs
        whiteness = self.whiteness_head(shared) * 100  # Scale to 0-100
        grade_logits = self.grade_head(shared)
        defect_prob = self.defect_head(shared)
        
        return whiteness.squeeze(), grade_logits, defect_prob.squeeze()


# ============================================
# 4. TRAINING PIPELINE
# ============================================

class PGModelTrainer:
    """Training pipeline for PG Quality Model."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.whiteness_criterion = nn.MSELoss()
        self.grade_criterion = nn.CrossEntropyLoss()
        self.defect_criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for images, whiteness, grades, defects in tqdm(dataloader, desc="Training"):
            images = images.to(self.device)
            whiteness = whiteness.to(self.device)
            grades = grades.to(self.device)
            defects = defects.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_whiteness, pred_grades, pred_defects = self.model(images)
            
            # Calculate losses (weighted)
            loss_w = self.whiteness_criterion(pred_whiteness, whiteness) * 0.4
            loss_g = self.grade_criterion(pred_grades, grades) * 0.4
            loss_d = self.defect_criterion(pred_defects, defects) * 0.2
            
            loss = loss_w + loss_g + loss_d
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct_grades = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, whiteness, grades, defects in dataloader:
                images = images.to(self.device)
                whiteness = whiteness.to(self.device)
                grades = grades.to(self.device)
                defects = defects.to(self.device)
                
                pred_whiteness, pred_grades, pred_defects = self.model(images)
                
                loss_w = self.whiteness_criterion(pred_whiteness, whiteness) * 0.4
                loss_g = self.grade_criterion(pred_grades, grades) * 0.4
                loss_d = self.defect_criterion(pred_defects, defects) * 0.2
                
                total_loss += (loss_w + loss_g + loss_d).item()
                
                # Grade accuracy
                _, predicted = torch.max(pred_grades, 1)
                correct_grades += (predicted == grades).sum().item()
                total_samples += grades.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_grades / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=30):
        print(f"Training on {self.device}")
        best_acc = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_pg_model.pth')
                print(f"  → Saved best model (acc: {best_acc:.4f})")
            
            self.scheduler.step()
        
        return self.history


# ============================================
# 5. INFERENCE CLASS (For Deployment)
# ============================================

class PGQualityInspector:
    """Production inference class for PG quality inspection."""
    
    GRADES = ['A', 'B', 'C', 'D']
    
    def __init__(self, model_path='best_pg_model.pth', device='cpu'):
        self.device = device
        self.model = PGQualityModel(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def inspect(self, image):
        """
        Inspect a PG sample image.
        
        Args:
            image: PIL Image or numpy array or file path
            
        Returns:
            dict with whiteness, grade, defect_detected, confidence
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            whiteness, grade_logits, defect_prob = self.model(img_tensor)
        
        # Process outputs
        whiteness_score = whiteness.item()
        grade_probs = torch.softmax(grade_logits, dim=1)[0]
        grade_idx = torch.argmax(grade_probs).item()
        confidence = grade_probs[grade_idx].item() * 100
        defect_detected = defect_prob.item() > 0.5
        
        # Determine pass/fail
        status = 'PASS' if grade_idx <= 1 else ('WARNING' if grade_idx == 2 else 'FAIL')
        
        return {
            'whiteness': round(whiteness_score, 1),
            'grade': self.GRADES[grade_idx],
            'grade_confidence': round(confidence, 1),
            'defect_detected': defect_detected,
            'defect_probability': round(defect_prob.item() * 100, 1),
            'status': status,
            'all_grade_probs': {g: round(p.item()*100, 1) for g, p in zip(self.GRADES, grade_probs)}
        }
    
    def inspect_batch(self, images):
        """Inspect multiple images."""
        return [self.inspect(img) for img in images]


# ============================================
# 6. MAIN EXECUTION
# ============================================

def main():
    # Configuration
    GENERATE_DATA = True  # Set False if you have real dataset
    DATA_DIR = './pg_dataset'
    NUM_SAMPLES = 1000
    BATCH_SIZE = 32
    EPOCHS = 20
    
    # Step 1: Generate/Load Dataset
    if GENERATE_DATA:
        metadata = SyntheticPGDataset.generate_dataset(NUM_SAMPLES, DATA_DIR)
    else:
        with open(f'{DATA_DIR}/metadata.json', 'r') as f:
            metadata = json.load(f)
    
    # Step 2: Train/Val Split
    train_meta, val_meta = train_test_split(metadata, test_size=0.2, random_state=42)
    
    # Step 3: Create Data Loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
    
    train_dataset = PGDataset(train_meta, f'{DATA_DIR}/images', transform)
    val_dataset = PGDataset(val_meta, f'{DATA_DIR}/images', val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Step 4: Create Model & Train
    model = PGQualityModel(pretrained=True)
    trainer = PGModelTrainer(model)
    history = trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Step 5: Plot Results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    # Step 6: Test Inference
    print("\n" + "="*50)
    print("Testing Inference...")
    inspector = PGQualityInspector('best_pg_model.pth')
    
    # Test on a few samples
    test_images = [f'{DATA_DIR}/images/A_0001.jpg', 
                   f'{DATA_DIR}/images/C_0001.jpg',
                   f'{DATA_DIR}/images/D_0001.jpg']
    
    for img_path in test_images:
        if os.path.exists(img_path):
            result = inspector.inspect(img_path)
            print(f"\n{img_path}:")
            print(f"  Whiteness: {result['whiteness']}%")
            print(f"  Grade: {result['grade']} ({result['grade_confidence']}% conf)")
            print(f"  Defect: {result['defect_detected']}")
            print(f"  Status: {result['status']}")


if __name__ == '__main__':
    main()