import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing




class method_text_encode:
    all_words = ["привет", "пока", "как", "дела", "хорошо", "плохо", "я", "рад", "не", "?", ",", ".", "у", "тоже", None, "нормально","норм", "это", "знаю", "погода", "везёт", "плохая", "ладно", "понял", "меня"]
    all_phrases = [
        {
            "привет":"привет",
            "как дела ?":"хорошо , у тебя?",
            "у меня тоже":"я рад.",
            "пока.":"пока.",
        },
        {
            "привет":"привет",
            "как дела ?":"хорошо , у тебя?",
            "у меня нормально":"это хорошо",
            "пока .":"пока .",
        },
        {
            "привет":"привет",
            "как дела ?":"хорошо , у тебя ?",
            "у меня нормально":"это хорошо",
            "какая погода у тебя ?":"хорошая",
            "тебе везёт , а у меня плохая": "понял",
            "пока":"пока",
        }
    ]
    MaxCount = 10
    MaxSeq = 7


    def __init__(self):
        self.matrix_words = preprocessing.normalize(np.random.randint(100, size=(1,len(self.all_words))))
        #self.matrix_words = np.random.randint(100, size=(1,len(self.all_words)))
        self.train_x = np.zeros((1, self.MaxSeq, self.MaxCount)).astype(np.float32)
        self.train_y = np.zeros((1, self.MaxSeq, self.MaxCount)).astype(np.float32)


    def encode_main(self):
        np.random.seed(1)
        for iter in self.all_phrases:
            data_iter_x = np.zeros((1, 10))
            data_iter_y = data_iter_x
            for x, y in iter.items():
                func_ = lambda d: [ self.matrix_words[0][self.all_words.index(i)] for i in d.split(" ") if i in self.all_words ]
                string_x = np.array(func_(x))
                string_y = np.array(func_(y))
                for iter_ in range(self.MaxCount - len(string_x)):
                    string_x = np.append(string_x, 0)
                for iter_ in range(self.MaxCount - len(string_y)):
                    string_y = np.append(string_y, 0)
                data_iter_x = np.vstack((data_iter_x, string_x))
                data_iter_y = np.vstack((data_iter_y, string_y))
            data_iter_x = np.delete(data_iter_x, 0, axis=0)
            data_iter_y = np.delete(data_iter_y, 0, axis=0)
            zero_ = np.zeros((1, 10))
            for i in range(self.MaxSeq - len(data_iter_x)):
                data_iter_x = np.vstack((data_iter_x, zero_))
            for i in range(self.MaxSeq - len(data_iter_y)):
                data_iter_y = np.vstack((data_iter_y, zero_))

            self.train_x = np.concatenate((self.train_x, np.array([data_iter_x])), axis=0)
            self.train_y = np.concatenate((self.train_y, np.array([data_iter_y])), axis=0)
    #######
    def __mini_encode(self, position, text):
        data_ = np.zeros((1, self.MaxCount))
        if position >= len(text):
            return data_[0].tolist()
        mini_text = list(filter(None, text[position].split(" ")))
        print(mini_text)
        for iter in range(len(mini_text)):
            symbols = mini_text[iter].lower()
            print(symbols)
            if symbols in self.all_words:
                index_ = self.all_words.index(symbols)
                print(self.matrix_words[0][index_])
                data_[0][iter] = self.matrix_words[0][index_]
        return data_[0].tolist()
    #######
    def text_in_matrix(self, text):
        main_data = np.array([[ self.__mini_encode(i, text) for i in range(self.MaxSeq) ]])
        #for i in range(self.MaxSeq):
        #    d = data[0][i]
        #    for iter in range(self.MaxCount-len(d)):
        return main_data
