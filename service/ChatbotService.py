import uuid

import pandas as pd
from sklearn.model_selection import train_test_split

from database import ChatbotRepository
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

    def getLabel(self):
        label = self.dataQuestion['qa_id']
        # chuyen doi nhan sang numeric
        d = dict(enumerate(label.unique(), 0))
        d = {value: key for key, value in d.items()}
        label = label.replace(d)
        return label

    def getLabel2(self, data):
        label = data['qa_id']
        # chuyen doi nhan sang numeric
        d = dict(enumerate(label.unique(), 0))
        d = {value: key for key, value in d.items()}
        label = label.replace(d)
        return label

    def getCategory(self):
        label = self.dataQuestion['category_id']
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
        data = self.dataQuestion
        categories = set(data['category_id'].to_list())
        print(categories)
        dt = data['question_content']
        label = self.getCategory()
        # trich xuat dac trung
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
                         [str(model_id), 'parent', Converter.encode(data), str(score), str(model.c),
                          Converter.encode(feature_extraction),
                          Converter.encode(model.wb)]);
        # child

        for e in categories:
            print(e)
            data_child = data[data['category_id'] == e]
            dt_child = data_child['question_content']
            label_child = self.getLabel2(data_child)
            # trich xuat dac trung
            feature_extraction_child = FeatureExtraction(dt_child, [])
            df_child = pd.DataFrame(data=feature_extraction_child.extract_to_array())

            X_train_child, X_test_child, y_train_child, y_test_child = self.train_test_split(df_child, label_child,
                                                                                             test_size=test_size,
                                                                                             random_state=random_state,
                                                                                             shuffle=shuffle)
            model_child = TrainingService.SVMModel(float(c), label_child)
            model_child.fit(X_train_child, y_train_child)
            model_child.train()
            score = model_child.score(X_test_child, y_test_child)
            model_id2 = uuid.uuid1()
            self.repo.insert("model", ['model_id', 'name', 'data', 'score', 'c_parameter', 'feature', 'wb'],
                             [str(model_id2), str(e), Converter.encode(data_child), str(score), str(model_child.c),
                              Converter.encode(feature_extraction_child),
                              Converter.encode(model_child.wb)]);
            self.repo.insert("model_child", ["model_child_id", "model_id"], [str(model_id2), str(model_id)])

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
        model = TrainingService.SVMModel()
        wb = self.chatbotModel[2]

        model.wb = Converter.decode(wb)
        return model.predict(content)

    def reply_question(self, question):
        answers = self.dataAnswer
        questions = self.dataQuestion['question_content']
        feature_extraction = FeatureExtractionService.FeatureExtractionService(questions, [])
        intention = self.predict(feature_extraction.tranform_new(question).toarray())
        return self.reply(intention, answers)

    def reply(self, intention, answers):
        print(intention)
        if intention == -1:
            return "Không tìm được câu trả lời phù hợp"
        d = dict(enumerate(self.dataQuestion['qa_id'].unique(), 0))
        d = {value: key for key, value in d.items()}
        key_list = list(d.keys())
        val_list = list(d.values())
        position = val_list.index(intention)
        print(answers[answers['qa_id'] == key_list[position]])
        return self.repo.getAnswerById(key_list[position])[0]
