import os
import re
import string
import math
import nltk
import random
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from functools import reduce

POS_TRIPLATES = {}
NEG_TRIPLATES = {}
NUM_FILES = 1000
TOTAL_POS = NUM_FILES
TOTAL_NEG = NUM_FILES
POS_DIR = []
NEG_DIR = []
ALL_DIR = []

def cleanReview(review):
    review = review.lower()
    stopWords = set(stopwords.words("english"))    
    
    words = [word for word in review.split(" ") if word not in stopWords and not re.match(".*\d+", word)]
    newReview = " ".join(words)

    stopWords |= set(["``", "''", "..."])
    tokenizer = RegexpTokenizer(r'\w+')
    tokenizedWords = tokenizer.tokenize(newReview)

    stemmedWords = []
    lemmatizer = WordNetLemmatizer()
    for word in tokenizedWords:
        stemmedWord = lemmatizer.lemmatize(word)
        if stemmedWord not in stopWords and stemmedWord not in string.punctuation and not stemmedWord.startswith("'") and len(stemmedWord) > 1:
            stemmedWords.append(stemmedWord)

    return stemmedWords

def wordsCounting(words, reviewName, isPos):
    counter = {}
    for word in words:
        if word in counter:
            counter[word] += 1
        else:
            counter[word] = 1
    
    triplets = []

    for key, val in counter.items():
        triplets.append([key, reviewName, val])

    return triplets

def writeTriplates():
    with open("triplets.txt", "w") as myfile:
        for filename, triplets in POS_TRIPLATES.items():
            for t in triplets:
                strTriplates = "(" + t[0] + "," + filename + "," + str(t[2]) + ")"
                myfile.write(strTriplates + "\n")
        for filename, triplets in NEG_TRIPLATES.items():
            for t in triplets:
                strTriplates = "(" + t[0] + "," + filename + "," + str(t[2]) + ")"
                myfile.write(strTriplates + "\n")        

def loadFiles():
    global POS_DIR
    global NEG_DIR
    global ALL_DIR
    # POS_DIR = os.listdir("movie_review_data/pos")[:NUM_FILES]
    POS_DIR = os.listdir("movie_review_data/pos")
    for filename in POS_DIR:
        with open("movie_review_data/pos/" + filename, "r") as myfile:
            filenamePure = filename.replace(".txt", "")
            ALL_DIR.append(["pos", filenamePure])
            words = cleanReview(myfile.read())
            triplets = wordsCounting(words, filenamePure, True)
            POS_TRIPLATES.setdefault(filenamePure, triplets)

    # NEG_DIR = os.listdir("movie_review_data/neg")[:NUM_FILES]
    NEG_DIR = os.listdir("movie_review_data/neg")
    for filename in NEG_DIR:
        with open("movie_review_data/neg/" + filename, "r") as myfile:
            filenamePure = filename.replace(".txt", "")
            ALL_DIR.append(["neg", filenamePure])
            words = cleanReview(myfile.read())
            triplets = wordsCounting(words, filenamePure, False)
            NEG_TRIPLATES.setdefault(filenamePure, triplets)

def calcMLE(trainingFiles):
    mle = {}
    wordSet = set()
    posTrainData = []
    negTrainData = []
    numPos = 0
    numNeg = 0
    for f in trainingFiles:
        if f[0] == "pos":
            posTrainData.extend(POS_TRIPLATES[f[1]])
            numPos += 1
        else:
            negTrainData.extend(NEG_TRIPLATES[f[1]])
            numNeg += 1

    for triplet in posTrainData:
        word, filename, count = triplet
        wordSet.add(word)
        if word not in mle:
            mle[word] = {count: {"pos": 1, "neg": 0}}
        else:
            if count not in mle[word]:
                mle[word][count] = {"pos": 1, "neg": 0}
            else:
                mle[word][count]["pos"] += 1

    for triplet in negTrainData:
        word, filename, count = triplet
        wordSet.add(word)
        if word not in mle:
            mle[word] = {count: {"pos": 0, "neg": 1}}
        else:
            if count not in mle[word]:
                mle[word][count] = {"pos": 0, "neg": 1}
            else:
                mle[word][count]["neg"] += 1
    return [mle, numPos, numNeg, len(wordSet)]

def classifier(review, triplates, mleResult):
    mle, numPos, numNeg, vocSize = mleResult
    
    piPos = []
    piNeg = []
    for triplate in triplates:
        word, filename, count = triplate
        if word in mle:
            if count in mle[word]:
                if "pos" in mle[word][count]:
                    if mle[word][count]["pos"] > 0:
                        prob = (mle[word][count]["pos"] + 1) / (numPos + vocSize)
                        piPos.append(math.log2(prob))
                    else:
                        prob =  1 / (numPos + vocSize)
                        piPos.append(math.log2(prob))
                if "neg" in mle[word][count]:
                    if mle[word][count]["neg"] > 0:
                        prob = (mle[word][count]["neg"] + 1) / (numNeg + vocSize)
                        piNeg.append(math.log2(prob))
                    else:
                        prob =  1 / (numNeg + vocSize)
                        piNeg.append(math.log2(prob))
            else:
                probPos = 1 / (numPos + vocSize)
                probNeg = 1 / (numNeg + vocSize)
                piPos.append(math.log2(probPos))
                piNeg.append(math.log2(probNeg))
        else:
            probPos = 1 / (numPos + vocSize)
            probNeg = 1 / (numNeg + vocSize)
            piPos.append(math.log2(probPos))
            piNeg.append(math.log2(probNeg))


    posProd = sum(piPos)
    negProd = sum(piNeg)
    
    result = "pos" if posProd > negProd else "neg"
    return result == review

def naiveBayes(portion):
    trainCount = math.floor(len(ALL_DIR) * portion)
    # trainCount = math.floor(NUM_FILES * portion)
    
    allDir = ALL_DIR
    random.shuffle(allDir)

    # allDir = allDir[:NUM_FILES]

    trainingFiles = allDir[:trainCount]
    testFiles = allDir[trainCount:]

    mleResult = calcMLE(trainingFiles)

    # test
    correct = 0
    wrong = 0

    for f in testFiles:
        result = False
        if f[0] == "pos":
            result = classifier(f[0], POS_TRIPLATES[f[1]], mleResult)
        else:
            result = classifier(f[0], NEG_TRIPLATES[f[1]], mleResult)
        if result:
            correct += 1
        else:
            wrong += 1

    correctRate = correct / (correct + wrong)
    return correctRate

print("Start Loading Files")
loadFiles()
print("Finish Loading Files")
writeTriplates()
portions = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
for i in range(len(portions)):
    print("Portion: " + str(portions[i]) + ": ")
    right = 0
    for j in range(5):
        result = naiveBayes(portions[i])
        right += result
    print("Corectness: " + "{0:.4f}".format(right / 5))
print("done")