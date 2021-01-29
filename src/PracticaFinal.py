# Import the required packages
import sklearn
import sklearn.datasets
import sklearn.model_selection as ns
import sklearn.discriminant_analysis
import sklearn.decomposition
import sklearn.neighbors
import sklearn.metrics
import numpy as np
import sklearn.cross_decomposition
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt


def calculaPCA(X, components):
    pca = sklearn.decomposition.PCA(n_components=components, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                                    iterated_power='auto', random_state=None)
    return pca.fit(X).transform(X)


def compute_test(x_test, y_test, clf, cv):
    splitter = sklearn.model_selection.KFold(n_splits=cv)
    indices = splitter.split(x_test, y_test)
    score = []
    for index_train, index_test in indices:
        X_test = x_test[index_test]
        Y_test = y_test[index_test]
        # clf.fit(X_train, Y_train)
        # clf.predict(X_test, Y_test)
        # clf.score(X_test, Y_test)
        score.append(accuracy_score(Y_test, clf.predict(X_test)))
    return score


def testIGrafica(clf, cv, X, Y):
    n_neighbours_max = 64
    n_dimensions_max = 64
    n_neighbours = range(1, n_neighbours_max)
    n_dimensions = range(1, n_dimensions_max)
    # clf = GridSearchCV(estimator=KNeighborsClassifier(), n_jobs=-1, param_grid={'n_neighbors': n_neighbours}, cv=cv)

    score = []
    for dimensions in n_dimensions:
        X_new = calculaPCA(X, dimensions)
        clf.fit(X_new, Y)
        score.append(clf.cv_results_['mean_test_score'])

    for i in n_neighbours:
        x = [i] * (n_dimensions_max - 1)
        plt.scatter(n_neighbours, x, c=score[i - 1])
    plt.title('Scatter plot GridCV')
    plt.xlabel('Neighbours')
    plt.ylabel('Dimensions')
    plt.colorbar()
    plt.show()


def KNNCerca(X, Y):
    X_train, X_test, Y_train, Y_test = ns.train_test_split(X, Y, test_size=0.3)
    # Generate possible values of the exploration Grid
    k = np.arange(128) + 1

    # Infer all the exploratory surface into parameters struct
    # parameters = {'n_neighbors' : k,
    #              'weights': ['uniform','distance'],
    #              'metric': ['euclidean', 'manhattan', 'minkowski']}
    parameters = {'n_neighbors': k}
    # Create Learner Factory
    knearest = sklearn.neighbors.KNeighborsClassifier()
    # Instantiate a GridSearch with the a) Learner Factory b) Exploratory parameters c) CrossValidation param

    clf = GridSearchCV(knearest, parameters, cv=10, n_jobs=-1)
    # Perform exploratory grid search over TrainingData
    clf.fit(X_train, Y_train)
    bestParam = clf.best_params_["n_neighbors"]
    mts = clf.cv_results_["mean_test_score"]
    testIGrafica(10, X, Y)
    return bestParam, mts


def NNCerca(X, Y):
    X_train, X_test, Y_train, Y_test = ns.train_test_split(X, Y, test_size=0.3)

    # Parameter creation
    # sizes=(np.arange(100)+1,np.arange(100)+1,)
    Nact = ["relu", "tanh", "logistic"]
    # parameters = {'hidden_layer_sizes' : sizes,'activation' : Nact, 'n_iter' : iteracions}
    parameters = {'hidden_layer_sizes': [(100,), (100, 100,), (100, 100, 100,)],
                  'activation': Nact}
    # 'n_iter_no_change': iteracions,
    # 'learning_rate': ['constant','adaptive']

    # NN creation
    nn = MLPClassifier(solver='sgd', learning_rate='constant', learning_rate_init=0.02)

    # Creem el gridSearch
    clf = GridSearchCV(nn, parameters, cv=10, n_jobs=-1)

    # Fem grid search
    clf.fit(X_train, Y_train)

    # Compute Test Accuracy with the already defined function (it has to be adapted)
    # score = compute_test(x_test=X_test,y_test = Y_test, clf = clf, cv =10)
    return clf


def adaBoostCerca(X, Y):
    X_train, X_test, Y_train, Y_test = ns.train_test_split(X, Y, test_size=0.3)

    parameters = {'n_estimators': np.arange(64) + 1}
    adaboost = AdaBoostClassifier()
    # Creem el gridSearch
    clf = GridSearchCV(adaboost, parameters, cv=10, n_jobs=-1)
    # Fit the data
    clf.fit(X, Y)
    # Compute Test Accuracy with the already defined function (it has to be adapted)
    # score = compute_test(x_test=X_test,y_test = Y_test, clf = clf, cv =10)
    bestParams = clf.best_params_
    mts = clf.cv_results_["mean_test_score"]
    return bestParams, mts


def plotArray(input, title):
    x = np.arange(len(input)) + 1
    plt.plot(x, input)
    plt.title(title)
    plt.xlabel('Parameters')
    plt.ylabel('score')
    plt.show()


def poltTrainTest(train, test):
    x = np.arange(len(train)) + 1
    plt.plot(x, train)
    x = np.arange(len(test)) + 1
    plt.plot(x, test)
    plt.title('Scatter plot GridCV')
    plt.xlabel('times')
    plt.ylabel('score')
    plt.show()


def newsGroup():
    # Importamos el dataset
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    # Vectorizamos
    print(newsgroups_train.DESCR)


if __name__ == "__main__":
    digits = sklearn.datasets.load_digits()
    X = digits.data
    Y = digits.target
    # veins, mts = cercaDeParametres(X, Y)
    # plotArray(mts, "KNN")
    newsGroup()
    # bestNN = NNCerca(X,Y)
    # a = bestNN.cv_results_["mean_test_score"]
    # plotArray(a, "DNN")
    # print(str(bestNN.best_params_))

    # bestAdaBoost, mtsAdaboost = adaBoostCerca(X,Y)
    # plotArray(mtsAdaboost, "AdaBoost")

    # X_train, X_test, Y_train, Y_test = ns.train_test_split(X, Y, test_size=0.3)
    # clf = AdaBoostClassifier(n_estimators=5)
    # Fit the data
    # clf.fit(X_train, Y_train)
    # score = accuracy_score(y_true=Y_test, y_pred=clf.predict(X_test), normalize=False)
    # print("Score: " + str(score) + " De: " + str(len(X_test)))
    # b = bestVeins.cv_results_["mean_test_score"]

    # print("Score:" + str(a))