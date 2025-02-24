import numpy as np
import sys

with open("cript2.txt", 'w') as f:
    sys.stdout = f
    num = 45

    def compute_p_im(pk, auth_tickets):
        """Вычисляет вероятности имитации p_им(x, a)."""
        n, r = auth_tickets.shape
        p_im = np.zeros((n, r))
        for x in range(n):
            for a in range(r):
                p_im[x, a] = sum(pk[k] for k in range(n) if auth_tickets[k, a] == auth_tickets[x, a])
        return p_im

    def compute_p_podm(pk, auth_tickets):
        """Вычисляет вероятности подмены p_подм(x’,a’;x,a)."""
        n, r = auth_tickets.shape
        p_podm = np.zeros((n, r, n, r))
        for x in range(n):
            for a in range(r):
                for xp in range(n):
                    if xp == x:
                        continue
                    for ap in range(r):
                        if auth_tickets[xp, ap] == auth_tickets[x, a]:
                            p_podm[xp, ap, x, a] = pk[xp]
        return p_podm

    def optimal_imitation_strategy(p_im):
        """Строит оптимальную стратегию имитации."""
        return np.argmax(p_im, axis=1)

    def optimal_substitution_strategy(p_podm):
        """Строит оптимальную стратегию подмены."""
        n, r, _, _ = p_podm.shape
        strategy = np.zeros((n, r, 2), dtype=int)  # Добавляем 2-мерный массив для хранения (xp, ap)
        
        for x in range(n):
            for a in range(r):
                xp, ap = np.unravel_index(np.argmax(p_podm[:, :, x, a]), (n, r))
                strategy[x, a] = [xp, ap]  # Теперь корректно записываем пару значений
        
        return strategy


    # Пример использования
    pk = np.array([0.25, 0.25, 0.25, 0.25])  # Равномерное распределение ключей для 4 ключей
    auth_tickets = np.array([
        [1, 2, 3],
        [2, 3, 1],
        [3, 1, 2],
        [1, 3, 2]
    ])

    p_im = compute_p_im(pk, auth_tickets)
    p_podm = compute_p_podm(pk, auth_tickets)
    opt_imit = optimal_imitation_strategy(p_im)
    opt_subst = optimal_substitution_strategy(p_podm)


    print("#"*num, "Задание 1", "#"*num)
    print(f"Вероятности имитации p_им(x, a):\n{p_im}\n")
    print(f"Вероятности подмены p_подм(x’,a’;x,a):\n{p_podm}\n")
    print(f"Оптимальная стратегия имитации:\n{opt_imit}\n")
    print(f"Оптимальная стратегия подмены:\n {opt_subst}\n")


    print("#"*num, "Задание 2", "#"*num)
    import numpy as np

    def compute_p_im(p_k, auth_tickets):
        """
        Вычисляет вероятности имитации p_им(x, a)
        """
        n, r = auth_tickets.shape
        p_im = np.zeros((n, r))
        
        for x in range(n):
            for a in range(r):
                p_im[x, a] = np.sum(p_k * (auth_tickets[:, a] == auth_tickets[x, a]))
        
        return p_im

    def compute_optimal_imitation(p_im):
        """
        Оптимальная стратегия имитации
        """
        return np.argmax(p_im, axis=1)

    def compute_p_podm(p_k, auth_tickets):
        """
        Вычисляет вероятности подмены p_подм(x’,a’;x,a)
        """
        n, r = auth_tickets.shape
        p_podm = np.zeros((n, r, n, r))
        
        for x in range(n):
            for a in range(r):
                for x_prime in range(n):
                    for a_prime in range(r):
                        if x_prime != x:
                            p_podm[x_prime, a_prime, x, a] = p_k[x_prime] * (auth_tickets[x_prime, a_prime] == auth_tickets[x, a])
        
        return p_podm

    def compute_optimal_substitution(p_podm):
        """
        Оптимальная стратегия подмены
        """
        n, r, _, _ = p_podm.shape
        optimal_substitution = np.zeros((n, r), dtype=int)
        
        for x in range(n):
            for a in range(r):
                x_prime, a_prime = np.unravel_index(np.argmax(p_podm[:, :, x, a]), (n, r))
                optimal_substitution[x, a] = x_prime  # или a_prime в зависимости от того, что требуется
        
        return optimal_substitution


    def generate_orthogonal_array(p):
        """
        Генерирует ортогональный массив (p, p+1, 1)-OA
        """
        oa = np.zeros((p, p + 1), dtype=int)
        for i in range(p):
            for j in range(p + 1):
                oa[i, j] = (i + j) % p
        return oa

    # Пример входных данных
    n, r = 3, 3  # Количество ключей и сообщений
    p_k = np.array([0.3, 0.5, 0.2])  # Распределение вероятностей ключей
    auth_tickets = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])  # Билеты аутентификации

    # Вычисления
    p_im = compute_p_im(p_k, auth_tickets)
    optimal_imitation = compute_optimal_imitation(p_im)
    p_podm = compute_p_podm(p_k, auth_tickets)
    optimal_substitution = compute_optimal_substitution(p_podm)

    # Ортогональный массив
    p_value = 3  # Простое число
    orthogonal_array = generate_orthogonal_array(p_value)

    # Вывод результатов
    print(f"Вероятности имитации p_им:\n{p_im}\n")
    print(f"Оптимальная стратегия имитации:\n{optimal_imitation}\n")
    print(f"Вероятности подмены p_подм:\n{p_podm}\n")
    print(f"Оптимальная стратегия подмены:\n{optimal_substitution}\n")
    print(f"Ортогональный массив (p, p+1, 1)-OA:\n{orthogonal_array}")



    sys.stdout = sys.__stdout__
