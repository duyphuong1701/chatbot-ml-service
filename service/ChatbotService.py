import uuid

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

from database import ChatbotRepository
from entity.Model import Model
from entity.PredictJsonResponse import PredictJsonResponse
from rest import RestClient
from service import FeatureExtractionService
from service import TrainingService
from util import Converter
from util.FeatureExtraction import FeatureExtraction


class ChatbotService:
    def __init__(self):
        self.repo = ChatbotRepository.Database()
        self.dataQuestion = self.repo.getAllQuestion()
        self.dataAnswer = self.repo.getAllAnswer()
        # self.chatbotModel = self.repo.getModel()

    def getLabel(self, y):
        label = y
        # chuyen doi nhan sang numeric
        d = dict(enumerate(label.unique(), 0))
        d = {value: key for key, value in d.items()}
        label = label.replace(d)
        return label

    def getLabel2(self, data):
        label = data['category']
        # chuyen doi nhan sang numeric
        d = dict(enumerate(label.unique(), 0))
        d = {value: key for key, value in d.items()}
        label = label.replace(d)
        return label

    def getCategory(self, data, field='category_id'):
        label = data[field]
        # chuyen doi nhan sang numeric
        d = dict(enumerate(label.unique(), 0))
        d = {value: key for key, value in d.items()}
        label = label.replace(d)
        return label

    def training1(self, c=5, random_state=1, shuffle=False, test_size=0.2):
        data = self.dataQuestion
        dt = data['question_content']
        label = self.getLabel()
        # tao bo trich xuat
        feature_extraction = FeatureExtraction(dt, [])
        df = pd.DataFrame(data=feature_extraction.extract_to_array())

        X_train, X_test, y_train, y_test = self.train_test_split(df, label, test_size=test_size,
                                                                 random_state=random_state, shuffle=shuffle)
        model = TrainingService.SVMModel(float(c), label)
        model.fit(X_train, y_train)
        model.train()
        score = model.score(X_test, y_test)
        model_id = uuid.uuid1()
        self.repo.insert("model", ['model_id', 'name', 'data', 'score', 'c_parameter', 'feature', 'wb'],
                         [str(model_id), 'sol1', Converter.encode(data), str(score), str(model.c),
                          Converter.encode(feature_extraction),
                          Converter.encode(model.wb)]);

    def train_test_split(self, df, label, random_state=1, shuffle=False, test_size=0.2):
        if random_state != None:
            return train_test_split(df, label, test_size=test_size, random_state=random_state, stratify=label)
        if shuffle == True:
            return train_test_split(df, label, test_size=test_size, shuffle=True, stratify=label)
        return train_test_split(df, label, test_size=0.2, shuffle=True, stratify=label)

    def training2(self, c=10, random_state=1, shuffle=False, test_size=0.2):
        print(f"Training with c={c},random_state= {random_state}, shuffle= {shuffle}, testsize={test_size}")
        data = Converter.toDataFrame(RestClient.getAllQuestion())
        data = data.sort_values(by=['group_id'], ascending=True)
        x = data['question_content']
        y1 = self.getLabel(data['group_id'])
        print(data['group_id'].unique())
        # trich xuat dac trung
        feature_extraction = FeatureExtraction(x, [])
        y2 = self.getLabel(data['category_id'])
        # data['category_id'] = y2
        # data['group_id'] = y1
        X_train, X_test, y_train, y_test = self.train_test_split(x, data[['category_id', 'group_id']],
                                                                 test_size=test_size,
                                                                 random_state=random_state, shuffle=shuffle)

        X_train = feature_extraction.extractFrom(X_train).toarray()
        model = self.train(X_train, y_train['group_id'], c, test_size)
        model.feature = feature_extraction
        score = model.score(feature_extraction.extractFrom(X_test).toarray(), y_test['group_id'])
        print(score)
        model_id = uuid.uuid1()
        model_parent = Model(model_id=str(model_id),
                             name='parent',
                             score=str(score),
                             c_parameter=str(model.c),
                             data=Converter.encode(model))
        list_model_save = []
        list_model_constraint = []
        # child
        list_child = []
        for e in data['group_id'].unique():
            print(f"Training child: {e}")
            data_child = data[data["group_id"] == e]
            x = data_child["question_content"]
            y = data_child["category_id"]
            feature_extraction_child = FeatureExtraction(x, [])
            x_feature_extraction_child = pd.DataFrame(data=feature_extraction_child.extract_to_array())
            model_child = TrainingService.SVMModel(c=float(c), label=self.getLabel(y))
            model_child.fit(x_feature_extraction_child, y)
            model_child.train()
            model_child.name = e
            model_child.feature = feature_extraction_child
            model_id2 = uuid.uuid1()
            model_child_save = Model(model_id=str(model_id2),
                                     name=str(e),
                                     score=str(None),
                                     c_parameter=str(model_child.c),
                                     data=Converter.encode(model_child))
            list_model_save.append(model_child_save)
            list_model_constraint.append(
                ["tb_model_child", ["model_child_id", "model_id"], [str(model_id2), str(model_id)]])
            list_child.append(model_child)
        model_parent.score = self.score_sol2(model, list_child, X_test, y_test['category_id'])
        list_model_save = [model_parent] + list_model_save
        for e in list_model_save:
            RestClient.postModel(e)
        for e in list_model_constraint:
            self.repo.insert(e[0], e[1], e[2])

    def score_sol2(self, model, list_child, x_test, y_test):
        y_predict = []
        for i in range(len(y_test)):
            feature_extraction_parent = model.feature
            group = model.predict(feature_extraction_parent.tranform_new(x_test.iloc[i]).toarray())
            if group != -1:
                y_predict_temp = []
                child_model = next((x for x in list_child if x.name == group), None)
                feature_extraction_child = child_model.feature
                print(feature_extraction_child.tranform_new(x_test.iloc[i]).toarray())
                category = child_model.predict(feature_extraction_child.tranform_new(x_test.iloc[i]).toarray())
                y_predict_temp.append(category)
                y_predict += list(y_predict_temp)
            else:
                y_predict.append(-1)
        print(list(y_test))
        print(y_predict)
        score = np.count_nonzero(y_test == y_predict) / len(y_test)
        return score

    def train(self, x, y, c, testsize):
        # label = self.getLabel(y)
        model = TrainingService.SVMModel(float(c), y)
        model.fit(x, y)
        model.train()
        return model

    def find_parameter_sol2(self, random_state=1):
        print(f"find parameter with sol2")
        c_list = [0.001, 0.05, 0.02, 0.1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
        for _c in c_list:
            print(f"c = {_c}")
            self.training2(c=_c, random_state=random_state, test_size=0.2)

    def find_parameter_sol1(self):
        print(f"find parameter with sol1")
        c_list = [0.001, 0.05, 0.02, 0.1, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
        for _c in c_list:
            self.training1(c=_c, random_state=1, test_size=0.2)

    def train_multi_sol2(self, c, times):
        print(f"train multi sol2 with c = {c} times = {times}")
        for x in range(times + 1):
            self.training2(c=c, shuffle=True, test_size=0.2, random_state=None)

    def train_multi_sol1(self, c, times):
        print(f"train multi sol1 with c = {c} times = {times}")
        for x in range(times + 1):
            self.training1(c=c, shuffle=True, test_size=0.2, random_state=None)

    def predict1(self, content):
        selection = RestClient.getSelection()
        _model = RestClient.getModel(selection["modelId"])
        model = TrainingService.SVMModel()
        model.wb = Converter.decode(_model["wb"])
        return model.predict(content)

    def reply_question1(self, question):
        answers = self.dataAnswer
        selection = RestClient.getSelection()
        _model = RestClient.getModel(selection["modelId"])
        questions = Converter.decode(_model["data"])["question_content"]
        feature_extraction = FeatureExtractionService.FeatureExtractionService(questions, [])
        intention = self.predict1(feature_extraction.tranform_new(question).toarray())
        return self.reply1(intention, answers)

    def reply1(self, intention, answers):
        if intention == -1:
            return "Không tìm được câu trả lời phù hợp"
        d = dict(enumerate(self.dataQuestion['qa_id'].unique(), 0))
        d = {value: key for key, value in d.items()}
        key_list = list(d.keys())
        val_list = list(d.values())
        position = val_list.index(intention)
        print(answers[answers['qa_id'] == key_list[position]])
        return self.repo.getAnswerById(key_list[position])[0]

    #     for sol2
    def reply_question2(self, question):
        category_id = self.predict2_find_label(question)
        # get via DB
        result = RestClient.getAnswerByCategoryId(category_id)
        print(result)
        return result

    def predict2_find_label(self, question):
        current_selection = RestClient.getSelection()
        group_model = Converter.decode(RestClient.getModel(current_selection['model_id'])['data'])
        feature_extraction_parent = group_model.feature
        list_child = []
        for x in current_selection['child']:
            list_child.append(Converter.decode(RestClient.getModel(x)['data']))

        group = group_model.predict(feature_extraction_parent.tranform_new(question).toarray())
        if group != -1:
            child_model = next((x for x in list_child if x.name == group), None)
            feature_extraction_child = child_model.feature
            category = child_model.predict(feature_extraction_child.tranform_new(question).toarray())
            return category
        return -1

    def predict2(self, model_id, question, flag="category"):
        model_db = RestClient.getModel(model_id)
        _questions = Converter.decode(model_db["data"])["question_content"]
        _group = Converter.decode(model_db["data"])["group_id"]
        _categories = Converter.decode(model_db["data"])["category_id"]
        _model_wb = Converter.decode(model_db["wb"])
        _feature_extraction = FeatureExtractionService.FeatureExtractionService(_questions, [])
        model = TrainingService.SVMModel()
        model.wb = _model_wb
        result = model.predict(_feature_extraction.tranform_new(question).toarray())
        if (flag == "group"):
            return (self.fromIndexToLabel(result, _group))
        return (self.fromIndexToLabel(result, _categories))

    def fromIndexToLabel(self, index, arr):
        return arr.unique()[index]

    def predict_json(self, data):
        current_selection = RestClient.getSelection()
        model = RestClient.getModel(current_selection['model_id'])
        model_child = []
        for x in current_selection['child']:
            model_child.append(Converter.decode(RestClient.getModel(x)['data']))
        result = self.score_sol2(model = Converter.decode(model['data']),list_child=model_child,x_test=data['question_content'],y_test=data['category_id'])
        return PredictJsonResponse(result)
