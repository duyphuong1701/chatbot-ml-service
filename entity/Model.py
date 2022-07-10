class Model:
    def __init__(self,model_id,name,data,score,c_parameter,feature,wb):
        self.model_id = model_id
        self.name = name
        self.data =data
        self.score=score
        self.c_parameter=c_parameter
        self.feature=feature
        self.wb=wb