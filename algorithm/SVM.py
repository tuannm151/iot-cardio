import numpy as np
import cvxopt
import cvxopt.solvers


class SupportVectorMachine:
    def __init__(self):
        self.w = None
        self.b = None
        self.X = None
        self.y = None
        self.C = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_coefficients = None

    def fit(self, X, y, C=1.0):
        """
        X: ma trận feature
        y: vector nhãn
        C: hệ số điều chỉnh
        """
        self.X = X
        self.y = y
        self.C = C
        n_samples, n_features = X.shape

        # tính ma trận kernel
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])

        # tính ma trận P
        P = cvxopt.matrix(np.outer(y, y) * K)
        # tính vector q
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # tính ma trận A
        A = cvxopt.matrix(y, (1, n_samples))
        # tính vector b
        b = cvxopt.matrix(0.0)
        # tính vector G
        G = cvxopt.matrix(
            np.vstack((np.eye(n_samples) * -1, np.eye(n_samples))))
        # tính vector h
        h = cvxopt.matrix(
            np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        # giải bài toán tối ưu
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # lấy các lagrange multipliers
        a = np.ravel(solution['x'])

        # tìm các support vectors
        # chỉ lấy các lagrange multipliers > 1e-5
        sv = a > 1e-5
        # chỉ lấy các lagrange multipliers > 1e-5
        ind = np.arange(len(a))[sv]
        # lấy các lagrange multipliers > 1e-5
        self.support_vector_coefficients = a[sv]
        # lấy các support vectors
        self.support_vectors = X[sv]
        # lấy các nhãn của các support vectors
        self.support_vector_labels = y[sv]

        # tính w

        self.w = np.zeros(n_features)
        for n in range(len(self.support_vector_coefficients)):
            self.w += self.support_vector_coefficients[n] * \
                self.support_vector_labels[n] * self.support_vectors[n]

        # tính b

        self.b = 0
        for n in range(len(self.support_vector_coefficients)):
            self.b += self.support_vector_labels[n]
            self.b -= np.sum(self.support_vector_coefficients *
                             self.support_vector_labels * K[ind[n], sv])
        self.b /= len(self.support_vector_coefficients)

    def project(self, X):

        return np.dot(X, self.w) + self.b

    def predict(self, X):

        return np.sign(self.project(X))

    def score(self, X, y):
        y_predict = self.predict(X)
        return np.mean(y_predict == y)
