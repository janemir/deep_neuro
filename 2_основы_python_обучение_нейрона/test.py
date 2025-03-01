# Мирошниченко 23ВП2

# Задание 1:
# 1 создайте список, заполненный случайными числами
# 2 создайте цикл, который проходит все элементы списка и суммирует только четные значения
# 3 выведите полученную сумму на экран в консоли

from random import randint

random_numbers = [randint(1, 10) for _ in range(10)]

print("Список случайных чисел:", random_numbers)

sum_even = 0  

for number in random_numbers:
    if number % 2 == 0:  
        sum_even += number  

print("Сумма четных чисел в списке:", sum_even)