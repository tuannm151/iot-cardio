import math


def predictKNN(X_train, y_train, arr_train, number):
    X = X_train
    y = y_train

    arr = []
    arr_res = []
    number_0 = 0
    number_1 = 0
    for index in range(len(X)):
        sum = 0
        for i in range(len(X[index])):
            sum += (X[index][i]-arr_train[i])**2
        dis = math.sqrt(sum)
        if len(arr_res) < number:
            arr.append(dis)
            arr_res.append(y[index])
        else:
            for index_2 in range(len(arr_res)):
                if arr[index_2] > dis:
                    arr[index_2] = dis
                    arr_res[index_2] = y[index]
                    break
    for arr_res_item in arr_res:
        if arr_res_item == 1:
            number_1 += 1
    res = {'1': float(number_1)/number*100,
           '0': 100-float(number_1)/number*100}
    out_come = 0
    if float(number_1)/number*100 > 50:
        out_come = 1
    return {'res': res,
            'out_come': out_come}


def scoreKNN(X_test, y_test, number):
    arr_res_test = []
    X = X_test
    y = y_test

    item_idex = 0
    for item in X:
        item_idex += 1
        arr_res_test.append(predictKN(
            X_test, y_test, item, number)['out_come'])
    res = 0
    for index in range(len(arr_res_test)):
        if arr_res_test[index] == y[index]:
            res += 1
    return (float(res)/len(arr_res_test))*100


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, arr_train):
        return predictKNN(self.X_train, self.y_train, arr_train, self.n_neighbors)

    def score(self, X_test, y_test):
        return scoreKNN(X_test, y_test, self.n_neighbors)
