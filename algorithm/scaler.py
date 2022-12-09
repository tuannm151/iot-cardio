import numpy as np
class Scaler:
    """
    Chuẩn hoá feature theo phương pháp chuẩn hoá theo trung bình và độ lệch chuẩn
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        """
        Tính trung bình và độ lệch chuẩn của dữ liệu
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return (data - self.mean) / self.std

    def transform(self, data):
        """
        Chuẩn hoá dữ liệu theo trung bình và độ lệch chuẩn đã tính
        """
        return (data - self.mean) / self.std
