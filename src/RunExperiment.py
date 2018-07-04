#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot
from model.mlp import MultilayerPerceptron


def main():
    for learningRateMillis in xrange(1, 30, 1):
        learningRate = learningRateMillis * 0.001
        for epochs in xrange(1, 60, 1):

            data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                              oneHot=True)

            myMultilayerPerceptronClassifier = MultilayerPerceptron(data.trainingSet,
                                                                    data.validationSet,
                                                                    data.testSet,
                                                                    learningRate=learningRate,
                                                                    epochs=epochs)

            evaluator = Evaluator()
            myMultilayerPerceptronClassifier.train()
            perceptronPred = myMultilayerPerceptronClassifier.evaluate()
            evaluator = Evaluator()

            print("\nResult of the Perceptron recognizer with learningRate {} and {} epochs:").format(learningRate, epochs)
            #evaluator.printComparison(data.testSet, perceptronPred)
            evaluator.printAccuracy(data.testSet, perceptronPred)

    

    
    
if __name__ == '__main__':
    main()
