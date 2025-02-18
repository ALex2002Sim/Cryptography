import sys
from sympy import randprime, Matrix, mod_inverse, gcd
import numpy as np

with open('cript1.txt', 'w') as f:
    sys.stdout = f
    num = 45

    print("#"*num, "Задание 1", "#"*num)

    k = 7
    size = 30 - k
    p = randprime(2**(size - 1), 2**size - 1)
    n = p
    print(f"k = {k}\np = n = {p}\n")

    def add(a, b, mod):
        return (a + b) % mod

    def mult(a, b, mod):
        return (a * b) % mod

    def opp(a, mod):
        return (mod - a) % mod
    
    def encrypt(u, M, n):
        return np.dot(u, M) % n

    mtr = np.array([[1241235, 2], [3, 4]], dtype=int)
    print("Исходная матрица:")
    print(mtr)

    det_mtr = int(np.linalg.det(mtr)) % n
    print(f"detM = {det_mtr}")

    inv_det = pow(det_mtr, -1, n)
    print(f"inv_det = {inv_det}\n")

    inv_mtr = np.zeros((2, 2), dtype=int)
    inv_mtr[0][0] = mult(mtr[1][1], inv_det, n)
    inv_mtr[0][1] = mult(opp(mtr[0][1], n), inv_det, n)
    inv_mtr[1][0] = mult(opp(mtr[1][0], n), inv_det, n)
    inv_mtr[1][1] = mult(mtr[0][0], inv_det, n)

    print(f"Обратная матрица по модулю {n}:")
    print(inv_mtr, '\n')

    identity_matrix = encrypt(mtr, inv_mtr, n)
    print(f"Результат умножения исходной матрицы на обратную по модулю {n}:")
    print(identity_matrix, "\n")

    u = np.array([21, 40])
    c = encrypt(u, mtr, n)
    y = encrypt(c, inv_mtr, n)

    print(f"Исходный вектор: {u}\nЗашифрованный: {c}\nРасшифрованный: {y}")

    print("#"*num, "Задание 2", "#"*num)

    u1 = np.array([21, 40])
    c1 = encrypt(u1, mtr, n)
    print(f"Пара 1: u1 = {u1}, c1 = {c1}")

    u2 = np.array([19, 41])
    c2 = encrypt(u2, mtr, n)
    print(f"Пара 2: u2 = {u2}, c2 = {c2}")

    # Решаем систему уравнений в модульной арифметике
    A = Matrix([
        [u1[0], u1[1], 0, 0],
        [0, 0, u1[0], u1[1]],
        [u2[0], u2[1], 0, 0],
        [0, 0, u2[0], u2[1]]
    ])

    b = Matrix([c1[0], c1[1], c2[0], c2[1]])

    # Решаем систему уравнений
    solution = A.solve(b)

    mtr_rec = A.inv_mod(n)*b % n
    mtr_rec = np.array([[mtr_rec[0], mtr_rec[2]], [mtr_rec[1], mtr_rec[3]]], dtype=int)

    print("\nВосстановленная матрица M:")
    print(mtr_rec)

    print("#"*num, "Задание 3", "#"*num)
    u1 = np.array([0, 1])
    u2 = np.array([1, 0])

    c1 = np.dot(u1, mtr) % n
    c2 = np.dot(u2, mtr) % n

    print("Атака по выбираемому открытому тексту:")
    print(f"u1 = {u1}, u2 = {u2}")
    print(f"c1 = {c1} (строка M: {mtr[1]})")
    print(f"c2 = {c2} (строка M: {mtr[0]})\n")

    det_M = int(np.linalg.det(mtr)) % n
    inv_det_M = pow(det_M, -1, n)

    mtr_inv = np.array([
        [mtr[1, 1] * inv_det_M % n, -mtr[0, 1] * inv_det_M % n],
        [-mtr[1, 0] * inv_det_M % n, mtr[0, 0] * inv_det_M % n]
    ])

    c1 = np.array([0, 1])
    c2 = np.array([1, 0])

    u1_inv_chosen = np.dot(c1, mtr_inv) % n
    u2_inv_chosen = np.dot(c2, mtr_inv) % n

    print("Атака по выбираемому шифртексту:")
    print(f"c1 = {c1}, c2 = {c2}")
    print(f"u1 = {u1_inv_chosen} (строка M^(-1): {mtr_inv[1]})")
    print(f"u2 = {u2_inv_chosen} (строка M^(-1): {mtr_inv[0]})")



    print("#"*num, "Задание 4", "#"*num)

    alphabet = "АБВГҐДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    char_to_num = {char: i for i, char in enumerate(alphabet)}
    num_to_char = {i: char for i, char in enumerate(alphabet)}

    def mod_inverse(a, m):
        for x in range(1, m):
            if (a * x) % m == 1:
                return x
        return None

    def affine_decrypt(ciphertext, a, b, m):
        a_inv = mod_inverse(a, m)
        if a_inv is None:
            return None
        
        plaintext = ""
        for char in ciphertext:
            if char in char_to_num:
                y = char_to_num[char]
                x = (a_inv * (y - b)) % m
                plaintext += num_to_char[x]
            else:
                plaintext += char

        return plaintext

    ciphertext_1 = "Ш Э К Ч Р Ч Ц Б К Э Д Ц Н Ц Н Э Ц Б Ч Ъ З О Э Ъ Э Ш Э С К Х"
    ciphertext_2 = "С Й О Ч Л У Д Ц Ф Ё Ь Р Ч Й Ю Е Ч Д Г Ю Е С Ё Р Й С Ё Я Ё Г"

    m = 33

    possible_a_values = [a for a in range(1, m) if gcd(a, m) == 1]
    possible_b_values = range(m)
    print(f"Возможные значения a: {possible_a_values}")
    print(f"Для каждого значения a будем перебирать следующие значения b: \n{np.array(possible_b_values)}\n")

    for a in possible_a_values:
        for b in possible_b_values:
            plaintext_1 = affine_decrypt(ciphertext_1, a, b, m)
            plaintext_2 = affine_decrypt(ciphertext_2, a, b, m)
            
            if plaintext_1 and plaintext_2:
                print(f"Попытка с a = {a}, b = {b}")
                print("Текст 1:", plaintext_1)
                print("Текст 2:", plaintext_2)
                print("-" * 70)

    
    
    

    sys.stdout = sys.__stdout__