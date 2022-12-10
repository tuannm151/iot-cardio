from utils.utils import corr_matrix
from db import DBConnection
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
import math
from utils.utils import train_test_split
from algorithm.scaler import Scaler
import time


class CardioML:
    conn = None
    df = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self):
        self.conn = DBConnection()
        df = pd.read_csv("cardio_train.csv", sep=";")
        self.df = self.cleaning(df)

    def cleaning(self, df):
        new_df = df.copy()
        # remove id column
        new_df.drop(columns="id", inplace=True)

        #  drop all duplicated rows
        new_df.drop_duplicates(inplace=True)
        return new_df

    def get_corr(self):
        corr_mat = corr_matrix(self.df.values)
        # lấy ma trận tương quan cho cột cardio
        corr_cardio = corr_mat[-1]
        result = []
        for i in range(len(corr_cardio)):
            result.append({
                "name": self.df.columns[i],
                "corr": corr_cardio[i]
            })
        return result

    def get_columns(self):
        return self.df.columns

    def save_record(self, algo, features, acc, sk_acc, test_size):
        # get current timestamp
        timestamp = int(time.time())
        # concat features
        features_str = ",".join(features)
        # insert to tblResult and tblSKLearn
        query = f"INSERT INTO tblResult (algo, features, acc, timestamp, test_size) VALUES ('{algo}', '{features_str}', {acc}, {timestamp}, {test_size})"
        sk_query = f"INSERT INTO tblSKLearn (algo, features, acc, timestamp, test_size) VALUES ('{algo}', '{features_str}', {sk_acc}, {timestamp}, {test_size})"

        self.conn.execute(query)
        self.conn.execute(sk_query)

    def scale_data(self):
        scaler = Scaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def split_train_test(self, features, test_size=0.2):
        X = self.df[features]
        y = self.df['cardio']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size)

    def logistic_regression(self):
        # sử dụng thuật toán tự xây dựng
        from algorithm.logisticRegression import LogisticRegression as myLogisticRegression
        my_lr = myLogisticRegression()
        # train model
        my_lr.fit(self.X_train, self.y_train)
        # get accuracy
        acc = my_lr.score(self.X_test, self.y_test)

        # sử dụng thuật toán sklearn
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(self.X_train, self.y_train)
        sk_acc = lr.score(self.X_test, self.y_test)

        return acc, sk_acc

    def decision_tree(self):
        from algorithm.decisionTree import DecisionTree as myDecisionTree
        my_dt = myDecisionTree(max_depth=2)
        my_dt.fit(self.X_train, self.y_train)
        acc = my_dt.score(self.X_test, self.y_test)

        from sklearn.tree import DecisionTreeClassifier
        dt = DecisionTreeClassifier(max_depth=2)
        dt.fit(self.X_train, self.y_train)
        sk_acc = dt.score(self.X_test, self.y_test)

        return acc, sk_acc

    def KNearestNeighbors(self):
        from algorithm.KNN import KNeighborsClassifier as myKNeighborsClassifier
        my_knn = myKNeighborsClassifier()
        my_knn.fit(self.X_train, self.y_train)
        acc = my_knn.score(self.X_test, self.y_test)

        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier()
        knn.fit(self.X_train, self.y_train)
        sk_acc = knn.score(self.X_test, self.y_test)

        return acc, sk_acc

    def Naive_Bayes(self):
        from algorithm.NaiveBayes import NaiveBayes as myNaiveBayes
        my_nb = myNaiveBayes()

        my_nb.fit(self.X_train, self.y_train)
        acc = my_nb.score(self.X_test, self.y_test)

        from sklearn.naive_bayes import GaussianNB
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        sk_acc = nb.score(self.X_test, self.y_test)

        return acc, sk_acc

    def SVM(self):
        from algorithm.SVM import SupportVectorMachine as mySVM
        my_svm = mySVM()
        my_svm.fit(self.X_train, self.y_train)
        acc = my_svm.score(self.X_test, self.y_test)

        from sklearn.svm import SVC
        svm = SVC()
        svm.fit(self.X_train, self.y_train)
        sk_acc = svm.score(self.X_test, self.y_test)

        return acc, sk_acc

    def execute_ml(self, algos, features, test_size):
        # split data
        self.split_train_test(features, test_size)
        # scale data
        self.scale_data()

        result = []

        # execute ml
        for algo in algos:
            acc = 0
            sk_acc = 0
            if algo == "logistic_regression":
                acc, sk_acc = self.logistic_regression()
            elif algo == "decision_tree":
                acc, sk_acc = self.decision_tree()
            elif algo == "knn":
                acc, sk_acc = self.KNearestNeighbors()
            elif algo == "naive_bayes":
                acc, sk_acc = self.Naive_Bayes()
            elif algo == "svm":
                acc, sk_acc = self.SVM()
            else:
                continue
            # save record
            self.save_record(algo, features, acc, sk_acc, test_size)
            result.append({
                "algo": algo,
                "acc": acc,
                "sk_acc": sk_acc
            })

        return result
