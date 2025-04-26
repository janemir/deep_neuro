import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import os
from PIL import Image
import shutil

# --- 1. Определение устройства ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# --- 2. Подготовка данных ---
def safe_rename_files(folder):
    """Безопасное переименование файлов с обработкой ошибок"""
    if not os.path.exists(folder):
        print(f"⚠️ Папка {folder} не существует!")
        return

    class_name = os.path.basename(folder)  # cats, cars или apples
    for i, filename in enumerate(os.listdir(folder)):
        try:
            # Пропускаем папки
            if os.path.isdir(os.path.join(folder, filename)):
                continue
                
            # Формируем новое имя
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png']:
                continue
                
            new_name = f"{class_name}_{i}{ext}"
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_name)
            
            # Переименовываем через копирование (надежнее)
            shutil.copy2(old_path, new_path)
            os.remove(old_path)
            print(f"✅ {filename} -> {new_name}")
            
        except Exception as e:
            print(f"❌ Ошибка с {filename}: {str(e)}")
            continue

# Обрабатываем все классы
for dataset_type in ['train', 'test']:
    for class_name in ['cats', 'cars', 'apples']:
        folder_path = os.path.join('custom_dataset', dataset_type, class_name)
        safe_rename_files(folder_path)

# --- 3. Загрузка данных ---
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = os.path.abspath('custom_dataset')

# Проверка структуры
print("\nПроверка структуры данных:")
train_cats_path = os.path.join(data_dir, 'train', 'cats')
test_apples_path = os.path.join(data_dir, 'test', 'apples')

print(f"Train cats: {len(os.listdir(train_cats_path))} файлов")
print(f"Test apples: {len(os.listdir(test_apples_path))} файлов")
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'train'),
    transform=data_transforms
)
test_dataset = datasets.ImageFolder(
    os.path.join(data_dir, 'test'),
    transform=data_transforms
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- 4. Модель ---
model = models.resnet18(pretrained=True)
num_classes = 3  # cats, cars, apples
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# --- 5. Обучение ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print("\nНачинаем обучение...")
for epoch in range(5):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Эпоха {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# --- 6. Проверка точности ---
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nТочность на тесте: {100 * correct / total:.2f}%")

# --- 7. Сохранение модели ---
torch.save(model.state_dict(), "custom_model.pth")
print("Модель сохранена как 'custom_model.pth'")