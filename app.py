from flask import Flask, request, jsonify
from flask_cors import CORS

from entity.ANS import respone
from service.ChatbotService import ChatbotService

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
    app.chatbotservice.train_multi_sol1(c=int(_c),times=int(_times))
    return "", 200
# for solution 2: div and conquer
@app.route('/train2', methods=['GET'])
def train2():
    app.chatbotservice.training2()
    return "", 200
@app.route('/find-parameter-sol2', methods=['GET'])
def find_parameter_sol2():
    app.chatbotservice.find_parameter_sol2()
    return "", 200
@app.route('/train-multi-sol2', methods=['GET'])
def train_multi_sol2():
    _c = request.args.get("c")
    _times = request.args.get("times")
    app.chatbotservice.train_multi_sol2(c=int(_c),times=int(_times))
    return "", 200
@app.route('/query', methods=['GET'])
def reply():
    question = request.args.get('question')
    ans = respone(app.chatbotservice.reply_question(question)).toJson()
    return ans

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765)