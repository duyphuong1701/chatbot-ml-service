class Model:
    def __init__(self,model_id,name,data,score,c_parameter):
        self.model_id = model_id
        self.name = name
        self.score=score
        self.c_parameter=c_parameter
        self.data = data
