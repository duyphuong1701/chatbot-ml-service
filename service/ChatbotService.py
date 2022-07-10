import uuid

import pandas as pd
from sklearn.model_selection import train_test_split

from database import ChatbotRepository
from entity.Model import Model
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

    def training2(self, c=5, random_state=1, shuffle=False, test_size=0.2):
        data = Converter.toDataFrame(RestClient.getAllQuestion())
        data = data.sort_values(by=['group_id'], ascending=True)
        x = data['question_content']
        y = self.getLabel(data['group_id'])
        print(data['group_id'].unique())
        # trich xuat dac trung
        feature_extraction = FeatureExtraction(x, [])
        x_feature_extraction = pd.DataFrame(data=feature_extraction.extract_to_array())

        X_train, X_test, y_train, y_test = self.train_test_split(x_feature_extraction, y, test_size=test_size,
                                                                 random_state=random_state, shuffle=shuffle)
        model = self.train(X_train, y_train, 10, 0.2)
        score = model.score(X_test, y_test)
        print(score)
        model_id = uuid.uuid1()
        model_save = Model(str(model_id), 'parent', Converter.encode(data), str(score), str(model.c),
                           Converter.encode(feature_extraction),
                           Converter.encode(model.wb))
        RestClient.postModel(model_save)
        # child

        for e in data['group_id'].unique():
            print(f"Training child: {e}")
            data_child = data[data["group_id"] == e]
            x = data_child["question_content"]
            y = self.getLabel(data_child["category_id"])
            feature_extraction = FeatureExtraction(x, [])
            x_feature_extraction = pd.DataFrame(data=feature_extraction.extract_to_array())
            model_child = TrainingService.SVMModel(float(c), y)
            model_child.fit(x_feature_extraction, y)
            model_child.train()
            model_id2 = uuid.uuid1()
            model_child_save = Model(str(model_id2), str(e), Converter.encode(data_child), str(score),
                                     str(model_child.c), Converter.encode(feature_extraction),
                                     Converter.encode(model_child.wb))
            RestClient.postModel(model_child_save)
            self.repo.insert("tb_model_child", ["model_child_id", "model_id"], [str(model_id2), str(model_id)])

    def train(self, x, y, c, testsize):
        label = self.getLabel(y)
        model = TrainingService.SVMModel(float(c), label)
        model.fit(x, y)
        model.train()
        return model

    def find_parameter_sol2(self):
        print(f"find parameter with sol2")
        c_list = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 5 * 10 ** 0, 10 ** 1, 10 ** 1 + 5, 2 * 10 ** 1, 5 * 10 ** 1,
                  10 ** 2, 10 ** 3]
        for _c in c_list:
            self.training1(c=_c, random_state=1, test_size=0.2)

    def find_parameter_sol1(self):
        print(f"find parameter with sol1")
        c_list = [10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 5 * 10 ** 0, 10 ** 1, 10 ** 1 + 5, 2 * 10 ** 1, 5 * 10 ** 1,
                  10 ** 2, 10 ** 3]
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
        selection = RestClient.getSelection()
        group_of_question = self.predict2(selection["model_id"], question, "group")
        category_id = None
        for child_id in selection["child"]:
            model_child = RestClient.getModel(child_id)
            if (model_child["name"] == group_of_question):
                category_id = self.predict2(child_id, question)
        result = RestClient.getAnswerByCategoryId(category_id)
        print(result)
        return result

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
