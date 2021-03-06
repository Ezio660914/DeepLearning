# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 06_NLP_1.py
@time: 2021-08-27 17:42
"""
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers.experimental.preprocessing import *
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pathlib
import datetime
from sklearn.model_selection import train_test_split

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")


class StaticConst:
    dataDir = pathlib.Path("resource/nlp_getting_started")
    trainDF = pd.read_csv(dataDir / "train.csv")
    testDF = pd.read_csv(dataDir / "test.csv")
    trainDFShuffled = trainDF.sample(frac=1)


def GetDataRandomly(dataFrame: pd.DataFrame, amount=1):
    randomIndex = random.randint(0, len(dataFrame) - amount)
    randomDF = dataFrame.sample(frac=1)
    for row in randomDF[["text", "target"]][randomIndex:randomIndex + amount].itertuples():
        _, text, target = row
        print("---------------------------------\n")
        print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
        print(f"Text:\n{text}\n")


def main():
    GetDataRandomly(StaticConst.trainDF)
    # use train test split to split training data into training and validation sets
    trainData, valData, trainLabel, valLabel = train_test_split(StaticConst.trainDFShuffled["text"].to_numpy(),
                                                                StaticConst.trainDFShuffled["target"].to_numpy(),
                                                                test_size=0.1)

    # Find average number of tokens (words) in training Tweets
    maxLength = round(sum([len(i.split()) for i in trainData]) / len(trainData))
    # manually set the max token length
    maxVocabLength = 10000
    # text vectorization, map word to integer vector
    textVectorizer = TextVectorization(max_tokens=maxVocabLength,
                                       output_mode="int",
                                       output_sequence_length=maxLength)
    # fit to the training text
    textVectorizer.adapt(trainData)
    # visualise the sentence with its vector
    sentence = random.choice(trainData)
    print(f"Original text:\n{sentence}\n\nVector:")
    print(textVectorizer([sentence]))
    # visualise the vocabulary of the textVectorizer
    vocab = textVectorizer.get_vocabulary()
    print(len(vocab))
    print(vocab[:5])
    print(vocab[-5:])

    pass


if __name__ == "__main__":
    main()
