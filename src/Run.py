#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from model.mlp import MultilayerPerceptron

from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot



def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=True)
    myStupidClassifier = StupidRecognizer(data.trainingSet,
                                          data.validationSet,
                                          data.testSet)
    
    myPerceptronClassifier = Perceptron(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)
                                        
    myLRClassifier = LogisticRegression(data.trainingSet,
                                        data.validationSet,
                                        data.testSet,
                                        learningRate=0.005,
                                        epochs=30)

    myMultilayerPerceptronData = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000,
                                                    oneHot=True)#other values on the Ex sheet?
    myMultilayerPerceptronClassifier = MultilayerPerceptron(myMultilayerPerceptronData.trainingSet,
                                                            myMultilayerPerceptronData.validationSet,
                                                            myMultilayerPerceptronData.testSet,
                                                            loss = 'bce',#evaluated: bce better than ce
                                                            learningRate=0.002,#good result
                                                            epochs=30)


    # Report the result #
    evaluator = Evaluator()                                        

    # Train the classifiers
    print("=========================")
    print("Note: Most debug output is supressed for performance. Can be changed in /model/mlp.py")
    print("=========================")
#other classifiers have been removed for performance.

    print("\nMultilayerPerceptron is training...")
    myMultilayerPerceptronClassifier.train()
    print("Done...")

    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()
    perceptronPred = myPerceptronClassifier.evaluate()
    lrPred = myLRClassifier.evaluate()
    mlpPred = myMultilayerPerceptronClassifier.evaluate()
    
    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("\nResult of the MultilayerPerceptron recognizer:")
    # evaluator.printComparison(data.testSet, mlpPred)
    evaluator.printAccuracy(data.testSet, mlpPred)
    
    # Draw -- DIMENSION ISSUE
    plot = PerformancePlot("MultilayerPerceptron validation")
    plot.draw_performance_epoch(myMultilayerPerceptronClassifier.performances,
                                myMultilayerPerceptronClassifier.epochs)
    
    
if __name__ == '__main__':
    main()
