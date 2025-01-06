

def edit_distance(seq1, seq2):
    # классический алгоритм Левенштейна
    n = len(seq1)
    m = len(seq2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m]

def calc_cer(target_text, predicted_text) -> float:
    # символный уровень
    target = list(target_text)
    pred = list(predicted_text)
    if len(target) == 0:
        # если target пустой, определим CER как 0 если pred тоже пустой иначе 1
        return 0.0 if len(pred) == 0 else 1.0
    dist = edit_distance(target, pred)
    return dist / len(target)

def calc_wer(target_text, predicted_text) -> float:
    # словарный уровень
    target = target_text.strip().split()
    pred = predicted_text.strip().split()
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    dist = edit_distance(target, pred)
    return dist / len(target)
