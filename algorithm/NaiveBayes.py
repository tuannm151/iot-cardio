import numpy as np

class NaiveBayes:
    def __init__(self):
        self.prior = None
        self.likelihood = None
        self.classes = None

    def fit(self, X, y):
        """
        X: ma trận feature
        y: vector nhãn
        """
        self.classes = np.unique(y)
        self.prior = np.zeros(len(self.classes))
        self.likelihood = np.zeros((len(self.classes), X.shape[1]))
        for i in range(len(self.classes)):
            self.prior[i] = np.sum(y == self.classes[i]) / len(y)
            self.likelihood[i] = np.sum(X[y == self.classes[i]], axis=0) / np.sum(y == self.classes[i])

    def predict(self, X):
        """
        X: ma trận feature
        y_pred: vector nhãn dự đoán
        """   
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = self.classes[np.argmax(np.log(self.prior) + np.sum(np.log(self.likelihood) * X[i], axis=1))]
        return y_pred

    def score(self, X, y):
        """
        X: ma trận feature
        y: vector nhãn
        accuracy: độ chính xác
        """
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
    
      
    

        
