from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtraction:
    def __init__(self, data, stopwords):
        self.data = data
        self.stopwords = stopwords
        self.vectorizer = self.__create_vector()

    def __create_vector(self):
        vectorizer = TfidfVectorizer(
            encoding='utf-8',
            stop_words=self.stopwords)
        return vectorizer.fit(self.data)

    def extract(self):
        return self.vectorizer.transform(self.data)

    def tranform_new(self, text):
        return self.vectorizer.transform([text])

    def extract_to_array(self):
        return self.extract().toarray()
