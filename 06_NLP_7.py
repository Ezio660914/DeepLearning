# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: 06_NLP_4.py
@time: 2021-08-28 21:19
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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


def EvaluateModel(y_true, y_pred):
    """
    calculate model accuracy, precision, recall and f1 score of a binary classification
    """
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    modelResult = {
        "accuracy": acc,
        "precision": pre,
        "recall": recall,
        "f1": f1
    }
    return modelResult


def main():
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
    # use tensorflow embedding layer to vectorize the text
    embedding = keras.layers.Embedding(input_dim=maxVocabLength,
                                       output_dim=128,
                                       input_length=maxLength)
    """create bidirectional RNN model"""
    inputs = keras.layers.Input(shape=(1,), dtype=tf.string)
    x = textVectorizer(inputs)
    x = embedding(x)
    x = keras.layers.Conv1D(filters=64,
                            kernel_size=5,
                            activation="relu",
                            padding="same")(x)
    x = keras.layers.GlobalMaxPool1D()(x)
    x = keras.layers.Dense(64, "relu")(x)
    outputs = keras.layers.Dense(1, "sigmoid")(x)
    model_5 = keras.Model(inputs, outputs)
    model_5.compile(keras.optimizers.Adam(),
                    keras.losses.BinaryCrossentropy(),
                    ["accuracy"])
    model_5.summary()
    history_5 = model_5.fit(trainData,
                            trainLabel,
                            epochs=5,
                            validation_data=(valData, valLabel))


if __name__ == "__main__":
    main()
