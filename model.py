"""
kstoeckl 694984
Constructs the model, trains it on the training data
and tests it on the validation data. 

Also has the capacity to hyper-parameter optimization, however this code is
currently commented out. With greater patience (greater search space,
both with regard to how fine the search is being done and also the 
parameters [For instance you could tinker with the learning rate of the 
SGDClassifer]) or with a better hyper-parameter optimization alg, perhaps
even greater accuracy could be achieved. 

Also generates the confusion_matrix graphic using code modified from
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py

"""
import pickle
import numpy as np
from time import time
from scipy.stats import mode
import matplotlib.pyplot as plt
from collections import Counter 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

X_TRAIN_FILE = "X_TRAIN.pkl"
Y_TRAIN_FILE = "Y_TRAIN.pkl"

X_TEST_FILE = "X_TEST.pkl"
Y_TEST_FILE = "Y_TEST.pkl"

#Found using Grid Search, on training data

#Optimal Parameters for Multinomial
OPT_ALPHA=0.204

#Optimal Parameters for SGDClassifier
OPT_ALPHA=0.0001
L1_RATIO = 0.3333
N_ITER = 50

def model(Xtrain,Ytrain,Xtest,Ytest):
    """
    Constructs and trains a multinomial Naive Bayes Learner on the training
    data, then produces performance metrics on the test data. 
    """
    model = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=OPT_ALPHA,
        l1_ratio=L1_RATIO, fit_intercept=True, n_iter=N_ITER, shuffle=True, verbose=0,
        epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal',
        eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)
    #model = MultinomialNB(alpha=OPT_ALPHA)
    model.fit(Xtrain,Ytrain)

    pred = model.predict(Xtest)

    i=0
    for a,b in zip(pred,Ytest):
        if i>20:
            break
        i+=1
        print(a,b)

    R0_Baseline = mode(Ytest)[1]/len(Ytest)
    print("Accuracy: ",metrics.accuracy_score(Ytest,pred))
    print(metrics.accuracy_score(Ytest,pred)*(len(Ytest)))
    print("Baseline: ",R0_Baseline," Guessing ",mode(Ytest)[0])
    print(metrics.classification_report(Ytest,pred))

    """
    #hyper-parameter optimization for MultinomialNB

    parameters = {'alpha':np.linspace(-10,10,50)}

    meta_opt = GridSearchCV(model, parameters)
    meta_opt.fit(Xtrain, Ytrain)

    print(meta_opt.best_params_)
    """     

    """
    #hyper-parameter optimization for SGDClassifier

    parameters = {'alpha':np.linspace(0.0001,10,10),
                'l1_ratio':np.linspace(0,1,10),
                'n_iter':np.linspace(50,200,4)}

    meta_opt = GridSearchCV(model, parameters,verbose=1)
    meta_opt.fit(Xtrain, Ytrain)

    print(meta_opt.best_params_)
    """
    return metrics.accuracy_score(Ytest,pred),Ytest,pred

def plot_confusion_matrix(cm,locations, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Modified From
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#example-model-selection-plot-confusion-matrix-py
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(locations))
    plt.xticks(tick_marks, locations, rotation=45)
    plt.yticks(tick_marks, locations)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__=="__main__":
    store = open(X_TRAIN_FILE, 'rb')
    Xtrain = pickle.load(store)    
    store.close()

    store = open(Y_TRAIN_FILE, 'rb')
    Ytrain = pickle.load(store)    
    store.close()

    store = open(X_TEST_FILE, 'rb')
    Xtest = pickle.load(store)    
    store.close()

    store = open(Y_TEST_FILE, 'rb')
    Ytest = pickle.load(store)    
    store.close()

    accuracy,yTest,yPred = model(Xtrain,Ytrain,Xtest,Ytest)
    cm = metrics.confusion_matrix(yTest,yPred)
    print(cm)

    locations = ["B","H","SD","Se","W"]

    plot_confusion_matrix(cm,locations)

    print(Counter(Ytrain))