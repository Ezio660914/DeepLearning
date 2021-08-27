# -*- coding: utf-8 -*-

"""
@author: fsy81
@software: PyCharm
@file: PreProcessFile.py
@time: 2021-08-27 10:23
"""
import shutil
import os
import threading

testTxtDir = "./resource/food101/food-101/meta/test.txt"
trainTxtDir = "./resource/food101/food-101/meta/train.txt"
txt = open(trainTxtDir, 'r', encoding='utf-8')
rootDir = "./resource/food101/food-101"
imageDir = "./resource/food101/food-101/images"


def CopyTrainSet():
    with open(trainTxtDir, 'r', encoding='utf-8') as txt:
        for img in txt.readlines():
            img = img[:-1]
            className = img.split("/")[0]
            src = imageDir + '/' + img + '.jpg'
            dstDir = rootDir + '/train/' + className
            os.makedirs(dstDir, exist_ok=True)
            dst = rootDir + '/train/' + img + '.jpg'
            shutil.copy(src, dst)
            print(f"copied {src} to {dst}")
    print("train set complete\n")


def CopyTestSet():
    with open(testTxtDir, 'r', encoding='utf-8') as txt:
        for img in txt.readlines():
            img = img[:-1]
            className = img.split("/")[0]
            src = imageDir + '/' + img + '.jpg'
            dstDir = rootDir + '/test/' + className
            os.makedirs(dstDir, exist_ok=True)
            dst = rootDir + '/test/' + img + '.jpg'
            shutil.copy(src, dst)
            print(f"copied {src} to {dst}")
    print("test set complete\n")


def main():
    t1 = threading.Thread(target=CopyTrainSet)
    t2 = threading.Thread(target=CopyTestSet)
    t1.start()
    t2.start()
    pass


if __name__ == "__main__":
    main()
