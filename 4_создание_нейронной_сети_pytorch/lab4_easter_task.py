import numpy as np
import pandas as pd

# 1. Загрузка данных
df = pd.read_csv('dataset_simple.csv')

# 2. Подготовка данных для линейной регрессии
X = np.hstack([np.ones((len(df), 1)), df[['age']].values])  # Добавляем столбец единиц для intercept
y = df['income'].values  # Предсказываем доход

# 3. Аналитическое решение линейной регрессии (нормальное уравнение)
w = np.linalg.inv(X.T @ X) @ X.T @ y

# 4. Вывод результатов
print("\nАналитическое решение линейной регрессии (пасхалка)")
print(f"Коэффициенты модели: intercept = {w[0]:.2f}, slope = {w[1]:.2f}")

# 5. Предсказания
predicted = X @ w
print("\nПервые 5 предсказаний:")
for i in range(5):
    print(f"Возраст: {df['age'].iloc[i]} лет | Реальный доход: {y[i]} | Предсказанный доход: {predicted[i]:.2f}")

# 6. Визуализация
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], y, color='blue', label='Реальные данные')
plt.plot(df['age'], predicted, color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('Возраст')
plt.ylabel('Доход')
plt.title('Линейная регрессия: предсказание дохода по возрасту')
plt.legend()
plt.grid(True)
plt.show()