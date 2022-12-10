# import dependencies
import matplotlib.pyplot as plt
import numpy as np


def corrcoef(x, y):
    """
    Tính toán hệ số tương quan giữa 2 cột của ma trận dữ liệu.
    Args:
        x: cột dữ liệu thứ nhất (n_samples x 1)
        y: cột dữ liệu thứ hai (n_samples x 1)
    Returns:
        Hệ số tương quan
    """
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sq = sum([pow(i, 2) for i in x])
    sum_y_sq = sum([pow(j, 2) for j in y])
    p_sum = sum([x[i] * y[i] for i in range(n)])
    num = p_sum - (sum_x * sum_y / n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) *
              (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0:
        return 0
    return num / den


def corr_matrix(data):
    """
    Tính toán ma trận tương quan giữa các cột của ma trận dữ liệu.
    Args:
        data: ma trận dữ liệu (n_samples x n_features)
    Returns:
        Ma trận tương quan (n_features x n_features)
    """
    n = len(data[0])
    corr_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                corr_mat[i][j] = 1.0
            else:
                corr = corrcoef(data[:, i], data[:, j])
                corr_mat[i][j] = corr
                corr_mat[j][i] = corr
    return corr_mat


def plot_corr_matrix(corr_mat, labels):
    """
    Vẽ ma trận tương quan.
    Args:
        corr_mat: ma trận tương quan (n_features x n_features)
        labels: tên các cột (n_features x 1)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_mat, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(labels), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.show()


def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1):
    """
    Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
    Args:
        X: dữ liệu đầu vào (n_samples x n_features)
        y: nhãn (n_samples x 1)
        shuffle: có xáo trộn dữ liệu hay không
        random_state: giá trị khởi tạo cho hàm xáo trộn
    """
    # convert dữ liệu về dạng numpy array
    _X = X
    _y = y
    if not isinstance(X, np.ndarray):
        _X = np.array(X)
    if not isinstance(y, np.ndarray):
        _y = np.array(y)
    
    # shuffle data
    if shuffle:
        _X = _X[np.random.RandomState(random_state).permutation(len(_X))]
        _y = _y[np.random.RandomState(random_state).permutation(len(_y))]
    # split data
    n = len(_X)
    n_train = int(n * (1 - test_size))
    X_train = _X[:n_train]
    y_train = _y[:n_train]
    X_test = _X[n_train:]
    y_test = _y[n_train:]

    return X_train, X_test, y_train, y_test


        
    