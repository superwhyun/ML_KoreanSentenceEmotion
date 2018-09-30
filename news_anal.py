
# 품사 코드 : https://docs.google.com/spreadsheets/d/1-9blXKjtjeKZqsf4NzHeYJCrr49-nXeRF6D80udfcwY/edit#gid=589544265
# mecab 형태소 분석기 : https://bitbucket.org/eunjeon/mecab-ko-dic/src/master/


# 
# 학습된 모델 저장하기 : pickle을 이용 (https://stackoverflow.com/questions/10017086/save-naive-bayes-trained-classifier-in-nltk)
#


import nltk as nltk
import json
from konlpy.tag import Mecab
import pickle


func=lambda x: 'pos' if x>0 else 'neg'


def term_exists(tokens, doc):
    # return {'exists({})'.format(word): (word in set(doc)) for word in tokens}
    # 학습시킬 때 exists라는 문자열이 들어가면 학습이 엉망으로 됨. 그래서 아래와 같이 고침
    return {'{}'.format(word): (word in set(doc)) for word in tokens}

def tokenize(pos_tagger, doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc)]

class NewsAnal:

    def __init__(self):
        self.mecab = Mecab()
        self.pos_tagger = self.mecab
        self.emo_dict_trained_model_fname = 'cfg/classifier.pickle'
        self.emo_dict_trained_token_fname = 'cfg/token.pickle'
        self.emo_dict_fname = 'data/SentiWord_info.json'
        pass


    def load_model(self):  

        ##############################
        # Load MODEL
        ##############################        
        try:
            fd = open(self.emo_dict_trained_model_fname, 'rb')
        except IOError:
            return False

        self.classifier = pickle.load(fd)
        if(self.classifier is None): return False
        fd.close()

        ##############################
        # Load TOKEN
        ##############################
        try:
            fd = open(self.emo_dict_trained_token_fname, 'rb')
        except IOError:
            return False

        self.tokens = pickle.load(fd)
        if(self.tokens is None): return False
        fd.close()

        return True

    def save_model(self):
        ##############################
        # save MODEL
        ##############################        
        fd = open(self.emo_dict_trained_model_fname, 'wb')
        pickle.dump(self.classifier, fd)
        fd.close()

        ##############################
        # save TOKEN
        ##############################
        fd = open(self.emo_dict_trained_token_fname, 'wb')
        pickle.dump(self.tokens, fd)
        fd.close()

        pass

    def load_data(self):
        with open('data/SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
	            data = json.load(f)
        return data


    def train(self, data):
        train = list()
        # 1000개 학습시킬때는 1~2초 걸렸는데, 14800개를 학습시키는데는 십분 가까이 걸린다. 왜 그럴까?
        for idx,item in enumerate(data):
            one = (item['word'], func(int(item['polarity'])))
            train.append(one)
            # if(idx > 1000): break  # unremark, just for testing

        train_docs = [(tokenize(self.pos_tagger, row[0]), row[1]) for row in train]
        self.tokens = [t for d in train_docs for t in d[0]]

        train_xy = [(term_exists(self.tokens, d), c) for d,c in train_docs]

        self.classifier = nltk.NaiveBayesClassifier.train(train_xy)
        self.classifier.show_most_informative_features() # 학습된 결과 보여주기

    def classify(self, sentence):
        test_docs = tokenize(self.pos_tagger, sentence)
        test_sentence_features = {word: (word in self.tokens) for word in test_docs}

        print("What U Typed :", sentence)
        print(test_sentence_features)
        result=self.classifier.classify(test_sentence_features)
        print("Your words are", result)



if __name__ == "__main__":

    na = NewsAnal()
    ret=na.load_model()

    if(ret is True):
        print('model loading completed')
        na.classifier.show_most_informative_features()
    elif(ret is False):
        print("trained model file is not found")
        print("begin training,.. wait")
        data=na.load_data()
        na.train(data)
        print("train completed and saved it into file")
        na.save_model()
    
    while(True):
        sentence = input("Input : ")    
        na.classify(sentence)










