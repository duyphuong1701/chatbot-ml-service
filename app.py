from flask import Flask, request
from flask_cors import CORS

from entity.ANS import respone
from service.ChatbotService import ChatbotService
from util import Converter

app = Flask(__name__)
app.chatbotservice = ChatbotService()
CORS(app)


@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


# for solution 1: normal
@app.route('/train1', methods=['GET'])
def train1():
    app.chatbotservice.training1()
    return "", 200


@app.route('/find-parameter-sol1', methods=['GET'])
def find_parameter_sol1():
    app.chatbotservice.find_parameter_sol1()
    return "", 200


@app.route('/train-multi-sol1', methods=['GET'])
def train_multi_sol1():
    _c = request.args.get("c")
    _times = request.args.get("times")
    app.chatbotservice.train_multi_sol1(c=int(_c), times=int(_times))
    return "", 200


# for solution 2: div and conquer
@app.route('/train2', methods=['POST'])
def train2():
    _c = float(request.values['c'])
    _random_state = int(request.values["random_state"])
    _shuffle = True if request.values["shuffle"] == 'True' else False
    _test_size = float(request.values["test_size"])
    app.chatbotservice.training2(c=_c, random_state=_random_state, shuffle=_shuffle, test_size=_test_size)
    return "", 200


@app.route('/find-parameter-sol2', methods=['POST'])
def find_parameter_sol2():
    _random_state = int(request.values["random_state"])
    app.chatbotservice.find_parameter_sol2(random_state=int(_random_state))
    return "", 200


@app.route('/train-multi-sol2', methods=['POST'])
def train_multi_sol2():
    _c = float(request.values['c'])
    _times = int(request.values["times"])
    app.chatbotservice.train_multi_sol2(c=int(_c), times=int(_times))
    return "", 200


@app.route('/query', methods=['GET'])
def reply():
    question = request.args.get('question')
    ans = respone(app.chatbotservice.reply_question1(question)).toJson()
    return ans


@app.route('/query2', methods=['GET'])
def reply2():
    question = request.args.get('question')
    lookp_dict = {"A": "diem_A",
                  "B": "diem_B",
                  "B+": "diem_B_cong",
                  "C": "diem_C",
                  "C+": "diem_C_cong",
                  "D": "diem_D",
                  "D+": "diem_D_cong",
                  "I": "diem_I",
                  "W": "diem_W",
                  "F": "diem_F",
                  "M": "diem_M",
                  }
    # performing split()
    temp = question.split()
    res = []
    for wrd in temp:
        # searching from lookp_dict
        res.append(lookp_dict.get(wrd, wrd))

    res = ' '.join(res)
    print(res)
    result = app.chatbotservice.reply_question2(res)
    return result, 200


@app.route('/predict-json', methods=['GET'])
def predictJson():
    question = request.get_json(silent=True)
    result = app.chatbotservice.predict_json(Converter.toDataFrame(question))
    return result.__dict__, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765)
