import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
df = pd.read_csv('dataset_simple.csv')
print(df.head())  # Проверяем структуру данных

# 2. Подготовка данных
# Используем правильные имена столбцов: age, income (признаки) и will_buy (целевая переменная)
X = df[['age', 'income']].values
y = df['will_buy'].values  # Должен содержать 0 (не купит) и 1 (купит)

# Нормализация данных 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Преобразуем в тензоры PyTorch
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Преобразуем в 2D тензор

# 3. Создание нейронной сети
class PurchasePredictor(nn.Module):
    def __init__(self):
        super(PurchasePredictor, self).__init__()
        self.layer1 = nn.Linear(2, 10)  # 2 входных признака
        self.layer2 = nn.Linear(10, 1)  # 1 выход (0 или 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

model = PurchasePredictor()

# 4. Функция потерь и оптимизатор
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. Обучение модели
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Эпоха [{epoch+1}/{epochs}], Ошибка: {loss.item():.4f}')

# 6. Оценка модели
with torch.no_grad():
    predicted = model(X)
    predicted = torch.round(predicted)  # Округляем до 0 или 1
    accuracy = (predicted == y).float().mean()
    print(f'Точность модели: {accuracy.item()*100:.2f}%')

# 7. Пример предсказания для новых данных
new_data = scaler.transform(np.array([[35, 50000], [25, 30000]]))
new_data = torch.tensor(new_data, dtype=torch.float32)
with torch.no_grad():
    print("\nПредсказания для новых данных (возраст, доход):")
    print("35 лет, 50000 ->", "Купит" if model(new_data[0]) > 0.5 else "Не купит")
    print("25 лет, 30000 ->", "Купит" if model(new_data[1]) > 0.5 else "Не купит")