import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier

domainlist = [] # store the domains

class Domain:
    def __init__(self, string, label = 1):
        self.length = len(string)
        self.Entropy = self.calEntropy(string)
        self.num = self.numbers(string)
        self.label = label

    def returnData(self):
        return [self.length, self.Entropy, self.num]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

    def calEntropy(self, string):
        entropy = 0.0
        sum_alpha = 0
        letter = np.zeros(26)
        string = string.lower()

        for i in range(len(string)):
            if string[i].isalpha():
                letter[ord(string[i]) - ord('a')] += 1
                sum_alpha += 1
        for i in range(26):
            p = 1.0 * letter[i] / sum_alpha
            if p > 0:
                entropy += -(p * math.log(p, 2))
        return entropy

    def numbers(self, string):
        sum_num = 0
        for i in range(len(string)):
            if string[i].isdigit():
                sum_num += 1
        return sum_num



def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            domainName = tokens[0]
            label = tokens[1]
            domainlist.append(Domain(domainName, label))


def main():
    initData("train.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)

    with open('test.txt') as f:
        with open("result.txt", 'w') as w:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                temp = clf.predict([Domain(line).returnData()])
                if temp == 0:
                    w.write(line+",notdga\n")
                else:
                    w.write(line+",dga\n")

if __name__ == '__main__':
    main()