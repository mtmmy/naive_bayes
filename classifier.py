import os
import re
import string
import math
import nltk
import random
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

POS_WORD_COUNTS = {}
NEG_WORD_COUNTS = {}
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

def wordsCounting(words, isPos):
    counter = {}
    for word in words:
        counter[word] = counter.setdefault(word, 0) + 1
    
    wordCounts = []
    for key, val in counter.items():
        wordCounts.append([key, val])

    return wordCounts

def writeTriplates():
    with open("triplets.txt", "w") as myfile:
        for filename, wordCounts in POS_WORD_COUNTS.items():
            for wordCount in wordCounts:
                strTriplates = "(" + wordCount[0] + "," + filename + "," + str(wordCount[1]) + ")"
                myfile.write(strTriplates + "\n")
        for filename, wordCounts in NEG_WORD_COUNTS.items():
            for wordCount in wordCounts:
                strTriplates = "(" + wordCount[0] + "," + filename + "," + str(wordCount[1]) + ")"
                myfile.write(strTriplates + "\n")        

def loadFiles():
    global POS_DIR
    global NEG_DIR
    global ALL_DIR
    POS_DIR = os.listdir("movie_review_data/pos")[:NUM_FILES]
    # POS_DIR = os.listdir("movie_review_data/pos")
    for filename in POS_DIR:
        with open("movie_review_data/pos/" + filename, "r") as myfile:
            filenamePure = filename.replace(".txt", "")
            ALL_DIR.append(["pos", filenamePure])
            words = cleanReview(myfile.read())
            wordCounts = wordsCounting(words, True)
            POS_WORD_COUNTS.setdefault(filenamePure, wordCounts)

    NEG_DIR = os.listdir("movie_review_data/neg")[:NUM_FILES]
    # NEG_DIR = os.listdir("movie_review_data/neg")
    for filename in NEG_DIR:
        with open("movie_review_data/neg/" + filename, "r") as myfile:
            filenamePure = filename.replace(".txt", "")
            ALL_DIR.append(["neg", filenamePure])
            words = cleanReview(myfile.read())
            wordCounts = wordsCounting(words, False)
            NEG_WORD_COUNTS.setdefault(filenamePure, wordCounts)

def calcMLE(trainingFiles):
    mle, wordSet = {}, set()
    posTrainData, negTrainData = [], []
    numPos, numNeg = 0, 0

    for f in trainingFiles:
        if f[0] == "pos":
            posTrainData.extend(POS_WORD_COUNTS[f[1]])
            numPos += 1
        else:
            negTrainData.extend(NEG_WORD_COUNTS[f[1]])
            numNeg += 1

    for wordCount in posTrainData:
        word, count = wordCount
        wordSet.add(word)
        if word not in mle:
            mle[word] = {count: {"pos": 1, "neg": 0}}
        else:
            if count not in mle[word]:
                mle[word][count] = {"pos": 1, "neg": 0}
            else:
                mle[word][count]["pos"] += 1

    for wordCount in negTrainData:
        word, count = wordCount
        wordSet.add(word)
        if word not in mle:
            mle[word] = {count: {"pos": 0, "neg": 1}}
        else:
            if count not in mle[word]:
                mle[word][count] = {"pos": 0, "neg": 1}
            else:
                mle[word][count]["neg"] += 1

    return [mle, numPos, numNeg, len(wordSet)]

def classifier(review, wordCounts, mleResult):
    mle, numPos, numNeg, vocSize = mleResult
    
    piPos, piNeg = [], []
    for wordCount in wordCounts:
        word, count = wordCount
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
                probPos, probNeg = 1 / (numPos + vocSize), 1 / (numNeg + vocSize)                
                piPos.append(math.log2(probPos))
                piNeg.append(math.log2(probNeg))
        else:
            probPos, probNeg = 1 / (numPos + vocSize), 1 / (numNeg + vocSize)
            piPos.append(math.log2(probPos))
            piNeg.append(math.log2(probNeg))

    posProd, negProd = sum(piPos), sum(piNeg)
    
    result = "pos" if posProd > negProd else "neg"
    return result == review

def naiveBayes(portion):
    # trainCount = math.floor(len(ALL_DIR) * portion)
    trainCount, allDir = math.floor(NUM_FILES * portion), ALL_DIR
    random.shuffle(allDir)

    allDir = allDir[:NUM_FILES]

    trainingFiles, testFiles = allDir[:trainCount], allDir[trainCount:]
    mleResult = calcMLE(trainingFiles)

    # test
    correct = 0

    for f in testFiles:
        result = False
        if f[0] == "pos":
            result = classifier(f[0], POS_WORD_COUNTS[f[1]], mleResult)
        else:
            result = classifier(f[0], NEG_WORD_COUNTS[f[1]], mleResult)
        if result:
            correct += 1

    correctRate = correct / (len(allDir) - trainCount)
    return correctRate

print("Start Loading Files")
loadFiles()
print("Finish Loading Files")
# writeTriplates()
portions = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
for i in range(len(portions)):
    print("Portion: " + str(portions[i]) + ": ")
    right = 0
    for j in range(5):
        result = naiveBayes(portions[i])
        right += result
    print("Corectness: " + "{0:.5f}".format(right / 5))
print("done")