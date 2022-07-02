# import gensim
# from pyvi import ViTokenizer

class Text:
    def __init__(self, str):
        self.str = str
        self.delTag()
        self.splitWord()

    def delTag(self):
        return None
        # # xoa tag
        # self.str = gensim.parsing.strip_tags(self.str)
        # # xoa khoang trang
        # self.str = gensim.parsing.preprocessing.strip_multiple_whitespaces(self.str)
        # # xoa ki tu dac biet
        # self.str = gensim.parsing.preprocessing.strip_non_alphanum(self.str)
        # # xoa dau , .
        # self.str = gensim.parsing.preprocessing.strip_punctuation(self.str)
        # # # xoa so
        # # self.str=gensim.parsing.preprocessing.strip_numeric(self.str)

    def splitWord(self):
        return None
        # self.str = ViTokenizer.tokenize(self.str)
