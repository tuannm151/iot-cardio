import numpy as np

class LogisticRegression:
    """
    Lớp LogisticRegression dùng để huấn luyện mô hình Logistic Regression phân loại dữ liệu
    Atrributes:
        learning_rate: tốc độ học
        iterations: số lần lặp
        weights: trọng số
        bias: hệ số điều chỉnh
    """
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        """
        Huấn luyện mô hình
        Args:
            X: dữ liệu đầu vào (ma trận n_samples x n_features)
            Y: dữ liệu đầu ra (ma trận n_samples x 1)
        """
        # khởi tạo trọng số và hệ số điều chỉnh
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # huấn luyện mô hình
        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            Y_predicted = self._sigmoid(linear_model)

            # tính toán đạo hàm của hàm loss
            dw = (1 / n_samples) * np.dot(X.T, (Y_predicted - Y))
            db = (1 / n_samples) * np.sum(Y_predicted - Y)

            # cập nhật trọng số và hệ số điều chỉnh
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        


        

    def predict(self, X):
        """
        Dự đoán kết quả
        Args:
            X: dữ liệu đầu vào (ma trận n_samples x n_features)
        Returns:
             ma trận dự đoán kết quả (ma trận n_samples x 1)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        Y_predicted = self._sigmoid(linear_model)
        Y_predicted_cls = [1 if i > 0.5 else 0 for i in Y_predicted]
        return np.array(Y_predicted_cls)
    
    def score(self, X, Y):
        """
        Tính toán độ chính xác của mô hình
        Args:
            X: dữ liệu đầu vào (ma trận n_samples x n_features)
            Y: dữ liệu đầu ra (ma trận n_samples x 1)
        Returns:
            độ chính xác của mô hình
        """
        Y_predicted = self.predict(X)
        # accuracy = tính tổng số giá trị dự đoán được đúng và chia cho tổng số giá trị
        return np.sum(Y_predicted == Y) / len(Y)

    def _sigmoid(self, x):
        """
        Hàm sigmoid dùng để chuyển đổi giá trị đầu vào thành giá trị đầu ra theo dạng hàm S (0 hoặc 1)
        Args:
            x: giá trị đầu vào
        Returns:
            giá trị đầu ra (0 hoặc 1)
        """
        return 1 / (1 + np.exp(-x))
    
    