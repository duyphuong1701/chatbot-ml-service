import numpy as np
import pandas as pd
import quadprog as qp
from tqdm import tqdm


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
    def __init__(self, c=None, label=None,wb=None,name=None,feature =None):
        self.c = c
        self.label = label
        self.wb = wb
        self.name =None
        self.feature =None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.label =  self.getLabel(y_train)
        self.arr = np.concatenate((np.array(X_train), np.array([y_train]).T), axis=1)

    def train(self):
        wb = []
        for j in tqdm(range(len(self.label.unique()))):
            # tao dictionary chuyen doi nhan
            d = dict(enumerate(np.full((len(self.label)), -1), 0))
            # doi nhan tuan tu
            d.update({j: 1})
            # goi SVM them vao mang luu
            x = SVM(self.X_train, self.getLabel(self.y_train).replace(d), self.c)
            x = x.sol()[0:x.n]
            wb.append(x)
        self.wb = np.array(wb)

    def _predict(self, input):
        wb = self.wb
        x_test = np.matrix(input)
        # thuc hien nhan xi*wi-b theo tung nhan
        v = (wb.T[:-1].T * x_test.T).T - wb[:, -1]
        # tim index max va phang hoa man tran ve 1d
        predict_ = np.ravel(np.argmax(v, axis=1))

        score =-np.sort(-v,axis=1)
        # print(np.min(score, axis=1).ravel())
        # score-=np.min(score, axis=1).ravel()
        idx = np.argsort(-v, axis=1)
        res=pd.DataFrame(data=[score.transpose().tolist(), idx.transpose().tolist()],index=['score',"index"]).transpose()
        # print(res)
        if np.max(score, axis=1).ravel()<0:
            return -1
        return predict_[0]
    def predict(self, input):
        index = self._predict(input)
        return self.y_train.unique()[index]

    def getLabel(self, y):
        label = y
        # chuyen doi nhan sang numeric
        d = dict(enumerate(label.unique(), 0))
        d = {value: key for key, value in d.items()}
        label = label.replace(d)
        return label

    def load_wb(self,filename):
        df = pd.read_csv(filename,header=None)
        self.wb = np.array(df)

    def score(self, X_test, y_test):
        wb = self.wb
        x_test = np.matrix(X_test)
        y_test = np.array(y_test)
        # thuc hien nhan xi*wi-b theo tung nhan
        v = (wb.T[:-1].T * x_test.T).T - wb[:, -1]
        # tim index max va phang hoa man tran ve 1d
        predict_ = np.ravel(np.argmax(v, axis=1))
        score = np.count_nonzero(y_test == predict_) / len(y_test)
        return score
# data = Converter.toDataFrame(RestClient.getAllQuestion())
# # data = data.iloc[:300,:]
# x = data["question_content"]
# y = data["group_id"]
# feature_extraction = FeatureExtraction(x, [])
# x_feature_extraction_child = pd.DataFrame(data=feature_extraction.extract_to_array())
# model = SVMModel(c=float(10))
# model.fit(x_feature_extraction_child, y)
# model.train()
# print(model.predict2(feature_extraction.tranform_new("thực tập thực tế").toarray()))