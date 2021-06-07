#!/usr/bin/python
# -*- coding: UTF-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Generación de diccionario

# Cargamos la data a modelar
text_1 = "El hombre fue a caminar"
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

from nltk.corpus import stopwords

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
