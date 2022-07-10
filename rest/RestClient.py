import json

import requests

from util import Constant

SELECTION_URL = "/selection"
GET_MODEL_BY_ID= "/model/{modelId}"
GET_ANSWER_BY_ID= "/answers/{id}/category"
GET_ALL_QUESTION = "/questions"
POST_MODEL = "/model"

def getSelection():
    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }
    data = requests.get(Constant.MANAGEMENT_SERVER+SELECTION_URL,headers=headers)
    return json.loads(data.content.decode())
def getModel(modelId):
    data = requests.get(Constant.MANAGEMENT_SERVER+GET_MODEL_BY_ID.format(modelId=modelId))
    return json.loads(data.content.decode())
def getAllQuestion():
    data = requests.get(Constant.MANAGEMENT_SERVER + GET_ALL_QUESTION)
    return data.json()
def postModel(model):
    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }
    requests.post(Constant.MANAGEMENT_SERVER+POST_MODEL, data=json.dumps(model.__dict__), timeout=10,headers=headers)
def getAnswerByCategoryId(categoryId):
    data = requests.get(Constant.MANAGEMENT_SERVER + GET_ANSWER_BY_ID.format(id=categoryId))
    return data.json()