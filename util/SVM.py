import csv
import uuid

import numpy as np
import pandas as pd
import quadprog as qp
from numpy import linalg
from tqdm import tqdm


def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))
class SVM:
    def __init__(self, x, y, c):
        self.c = c
        self.wb = []
        self.x = x
        self.y = y
        self.arr = np.concatenate((np.array(x), np.array([y]).T), axis=1)
        self.m, self.n = self.arr.shape
        self.G = self.getG()
        self.a = self.geta()
        self.C = self.getC()
        self.b = self.getb()

    def getG(self):
        G = np.zeros((self.n + self.m, self.n + self.m), dtype=float)
        u = np.full((1, self.n - 1), 1, dtype=float)
        v = np.full((1, self.m + 1), 0.0001, dtype=float)
        np.fill_diagonal(G, np.hstack((u, v)))
        return G

    def geta(self):
        u = np.ravel(np.zeros((1, self.n - 1), dtype=float))
        v = np.ravel(np.full((1, self.m + 1), -self.c, dtype=float))
        a = np.hstack((u, v))
        return a

    def getC(self):
        tempA11 = self.arr[:, 0:-1].T
        tempA12 = np.zeros((self.m, self.m), dtype=float)
        np.fill_diagonal(tempA12, self.arr[:, -1])
        tempA1 = np.dot(tempA11, tempA12).T
        tempA2 = np.zeros((self.m, self.n - 1), dtype=float)
        tempB1 = -1 * np.matrix(self.arr[:, self.n - 1]).T
        tempB2 = (np.zeros((self.m, 1), dtype=float))
        tempA = np.vstack((tempA1, tempA2))
        tempB = np.vstack((tempB1, tempB2))
        tempC = np.vstack((np.identity(self.m), np.identity(self.m)))
        C = np.hstack((tempA, tempB, tempC))
        return C.T

    def getb(self):
        u = np.ravel(np.full((1, self.m), 1, dtype=float))
        v = np.ravel(np.full((1, self.m), 0, dtype=float))
        b = np.hstack((u, v))
        return b

    def sol(self):
        solve = qp.solve_qp(self.G, self.a, self.C, self.b)
        wb = solve[0]
        return wb


class SVMModel:
    def __init__(self, c=None, label=None):
        self.c = c
        self.label = label
        self.wb = []

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.arr = np.concatenate((np.array(X_train), np.array([y_train]).T), axis=1)

    def train(self):
        wb = []
        for j in tqdm(range(len(self.label.unique()))):
            # tao dictionary chuyen doi nhan
            d = dict(enumerate(np.full((len(self.label)), -1), 0))
            # doi nhan tuan tu
            d.update({j: 1})
            # goi SVM them vao mang luu
            x = SVM(self.X_train, self.y_train.replace(d), self.c)
            x = x.sol()[0:x.n]
            wb.append(x)
        self.wb = np.array(wb)

    def predict(self, input):
        wb = self.wb
        x_test = np.matrix(input)
        # thuc hien nhan xi*wi-b theo tung nhan
        v = (wb.T[:-1].T * x_test.T).T - wb[:, -1]
        # tim index max va phang hoa man tran ve 1d
        predict_ = np.ravel(np.argmax(v, axis=1))

        score =-np.sort(-v,axis=1)
        print(np.min(score, axis=1).ravel())
        # score-=np.min(score, axis=1).ravel()
        idx = np.argsort(-v, axis=1)
        res=pd.DataFrame(data=[score.transpose().tolist(), idx.transpose().tolist()],index=['score',"index"]).transpose()
        print(res)
        if np.max(score, axis=1).ravel()<0:
            return -1
        return predict_[0]

    def save_csv(self,id,value):
        if id == None:
            id = uuid.uuid1()
        # opening the csv file in 'w+' mode
        filename = '../csv/history_train/training_' + str(id) + '.csv'
        file = open(filename, 'w+', newline='')

        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows(value)

    def save_wb(self,id):
        if id == None:
            id = uuid.uuid1()
        filename = '../csv/model/model_' + str(id) + '.csv'
        file = open(filename, 'w+', newline='')

        # writing the data into the file
        with file:
            write = csv.writer(file)
            write.writerows(self.wb)
    def load_wb(self,filename):
        df = pd.read_csv(filename,header=None)
        self.wb = np.array(df)

    def score(self, X_test, y_test):
        wb = self.wb
        print(wb)
        x_test = np.matrix(X_test)
        y_test = np.array(y_test)
        # thuc hien nhan xi*wi-b theo tung nhan
        v = (wb.T[:-1].T * x_test.T).T - wb[:, -1]
        # tim index max va phang hoa man tran ve 1d
        predict_ = np.ravel(np.argmax(v, axis=1))
        score = np.count_nonzero(y_test == predict_) / len(y_test)
        return score
