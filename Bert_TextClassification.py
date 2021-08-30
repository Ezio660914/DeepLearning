# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: Bert_TextClassification.py
@time: 2021-08-29 13:08
"""

import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import pandas as pd
import pathlib
import datetime
from official.nlp import optimization

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    print("GPU error")


class StaticConst:
    trainDir = pathlib.Path("resource/aclImdb/train")
    testDir = pathlib.Path("resource/aclImdb/test")
    batchSize = 32
    seed = 42
    bertModelDir = pathlib.Path("savedModel/small_bert_bert_en_uncased_L-8_H-512_A-8_2")
    bertPreprocessDir = pathlib.Path("savedModel/bert_en_uncased_preprocess_3")


def BuildClassifierModel():
    textInput = tf.keras.layers.Input(shape=(), dtype=tf.string)
    encodedInput = hub.KerasLayer(str(StaticConst.bertPreprocessDir))(textInput)
    encodedOutput = hub.KerasLayer(str(StaticConst.bertModelDir), trainable=True)(encodedInput)
    net = encodedOutput["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1)(net)
    net = tf.keras.layers.Activation(tf.keras.activations.sigmoid, dtype=tf.float32)(net)
    model = tf.keras.Model(textInput, net)
    return model


def main():
    # get the train dataset
    trainDs = tf.keras.preprocessing.text_dataset_from_directory(str(StaticConst.trainDir),
                                                                 batch_size=StaticConst.batchSize,
                                                                 validation_split=0.2,
                                                                 subset="training",
                                                                 seed=StaticConst.seed)
    classNames = trainDs.class_names
    trainDs = trainDs.cache().prefetch(tf.data.AUTOTUNE)

    # get a validation dataset
    valDs = tf.keras.preprocessing.text_dataset_from_directory(str(StaticConst.trainDir),
                                                               batch_size=StaticConst.batchSize,
                                                               validation_split=0.2,
                                                               subset="validation",
                                                               seed=StaticConst.seed)
    valDs = valDs.cache().prefetch(tf.data.AUTOTUNE)

    # get the test dataset
    testDs = tf.keras.preprocessing.text_dataset_from_directory(str(StaticConst.testDir),
                                                                batch_size=StaticConst.batchSize)
    testDs = testDs.cache().prefetch(tf.data.AUTOTUNE)

    """view some train examples"""
    # print("Training Examples:")
    # for textBatch, labelBatch in trainDs.take(1):
    #     for i in range(3):
    #         print(f"Review: {textBatch.numpy()[i]}")
    #         label = labelBatch.numpy()[i]
    #         print(f"Label: {label} ({classNames[label]})")
    # print("\n")

    # load preprocess model
    bertPreprocessModel = hub.KerasLayer(str(StaticConst.bertPreprocessDir))

    # try the preprocess model on some text
    textTest = ["this is such an amazing movie!"]
    textTestPreprocessed = bertPreprocessModel.call(textTest)
    print(f"Keys: {list(textTestPreprocessed.keys())}")
    print(f"Shape: {textTestPreprocessed['input_word_ids'].shape}")
    print(f"Word Ids: {textTestPreprocessed['input_word_ids'][0, :12]}")
    print(f"Input Mask: {textTestPreprocessed['input_mask'][0, :12]}")
    print(f"Type Ids: {textTestPreprocessed['input_type_ids'][0, :12]}")

    # load bert model
    bertModel = hub.KerasLayer(str(StaticConst.bertModelDir))
    # try the model on preprocessed text
    textTestResult = bertModel.call(textTestPreprocessed)

    print("\n-------Bert Outputs-------\n")
    print(f"Keys:{textTestResult.keys()}")
    print(f"Loaded Bert: {StaticConst.bertPreprocessDir.name}")
    print(f"Pooled Outputs Shape: {textTestResult['pooled_output'].shape}")
    print(f"Pooled Outputs Values: {textTestResult['pooled_output'][0, :12]}")
    print(f"Sequence Outputs Shape: {textTestResult['sequence_output'].shape}")
    print(f"Sequence Outputs Values: {textTestResult['sequence_output'][0, :12]}")

    # create model
    model = BuildClassifierModel()

    # define loss, metric, optimizer
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = tf.keras.metrics.BinaryAccuracy()
    epochs = 5
    stepsPerEpoch = tf.data.experimental.cardinality(trainDs).numpy()
    numTrainSteps = stepsPerEpoch * epochs
    numWarmupSteps = int(0.1 * numTrainSteps)
    initLearningRate = 3e-5
    optimizer = optimization.create_optimizer(init_lr=initLearningRate,
                                              num_train_steps=numTrainSteps,
                                              num_warmup_steps=numWarmupSteps,
                                              optimizer_type="adamw")
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    model.summary()
    history = model.fit(trainDs,
                        validation_data=valDs,
                        epochs=epochs)
    model.evaluate(testDs)


if __name__ == "__main__":
    main()
