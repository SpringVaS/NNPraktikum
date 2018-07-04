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
    results = []

    for learningRateMillis in xrange(1, 50, 5):
        learningRate = learningRateMillis * 0.001
        for epochs in xrange(1, 5, 1):

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

            results.append((learningRate, epochs, evaluator.getAccuracy(data.testSet, perceptronPred)))

    sortedResults = sorted(results, key = lambda x: x[2], reverse=True)

    for (learningRate, epochs, accuracy) in sortedResults:
        print("Learning Rate: {} Epochs: {} Accuracy: {}").format(learningRate, epochs, accuracy)

    

    
    
if __name__ == '__main__':
    main()
