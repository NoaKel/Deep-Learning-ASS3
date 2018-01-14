import sys
import pickle
from bilstmTrain import biLSTMModel

class data:
    def __init__(self,  file):
        self.parse_data(file)

    def parse_data(self, file):
        self.data = []
        sentence = []
        self.vocab = set()
        self.chars = set()
        self.suff = set()
        self.pref = set()

        for i, line in enumerate(open(file)):
            if line == '\n':
                self.data.append(sentence)
                sentence = []
            else:
                word = line[:-1]
                sentence.append(word)
                #if is_emb and (word not in self.vocab):
                self.vocab.add(word)
                #if is_LSTM:
                for c in word:
                    self.chars.add(c)
                #if is_sub:
                if word[:3] not in self.pref:
                    self.pref.add(word[:3])
                if word[-3:] not in self.suff:
                    self.suff.add(word[-3:])

if __name__ == "__main__":
    repr = sys.argv[1]
    modelFile = sys.argv[2]
    testFile = sys.argv[3]
    dictFile = sys.argv[4]
    outFile = sys.argv[5]

    testData = data(testFile)

    dataTrain = pickle.load(open(dictFile, "rb"))

    model = biLSTMModel(dataTrain, repr)
    model.load(modelFile)

    prediction = [model.forward(sentence) for sentence in testData.data]

    out = []
    for sentence, tags in zip(testData.data, prediction):
        for word, tag in zip(sentence, tags):
            out.append(word + " " + tag+ "\n")
        out.append("\n")

    f = open(outFile, "w")
    f.write("".join(out))
    f.close()