#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# Generación de diccionario

# Cargamos la data a modelar
text_1 = "El hombre fue a caminar"                  # Pasarlo como en R Data$Columna[n]
text_2 = "Los niños se sentaron sobre el cesped"

# Vectorizamos las palabras, separando por espacio
bagOfWordsA = text_1.split(' ')
bagOfWordsB = text_2.split(' ')

# Definir las palabras únicas
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
print(uniqueWords)

# Creación de diccionario para las palabras únicas.
# el diccionario se genera con las palabras de la data.

numOfWordsA = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsA:
    numOfWordsA[word] += 1

numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1

print(numOfWordsA)
print(numOfWordsB)

stopwords.words('spanish')


# TERM FRECUENCY (TF)

# Número total de veces que aparece una palabra en un texto, dividido por
# el total de palabras en el documento.
# Representa el valor de la celda en la matriz.

# obs: cada documento tiene su propia frecuencia.

def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


# Intervalos inferiores para los datas
tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)


# INVERSE DATA FRECUENCY (IDF)

# IDF determina la relevancia de las palabras únicas en todos los documentos.

def computeIDF(documents):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


# IDF para todos los documento
idfs = computeIDF([numOfWordsA, numOfWordsB])


# TF IDF

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf


# Aplicamos a Bases A y B
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

# Generación de matriz
df = pd.DataFrame([tfidfA, tfidfB])
print(df)

# MODELO TF IDF 'Suavizado' (incluye todas las palabras sin stopwords)

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform([text_1, text_2])

feature_names = vectorizer.get_feature_names()

dense = vectors.todense()

denselist = dense.tolist()

df = pd.DataFrame(denselist, columns=feature_names)
print(df)