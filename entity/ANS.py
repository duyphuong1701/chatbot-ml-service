import json


class respone:
    def __init__(self, answer):
        self.answer = answer

    def toJson(self):
        return json.dumps(self.__dict__,ensure_ascii=False).encode('utf-8')