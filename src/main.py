import numpy
import sklearn
import sklearn.datasets
import sklearn.model_selection as ns
import sklearn.discriminant_analysis
import sklearn.decomposition
import sklearn.neighbors
import sklearn.metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def calculaEstadistiques(X, Y):
    # Busquem el nombre de elements de entrenament per classe
    exemplesPerClasse = numpy.zeros(10)
    for i in range(0, Y.shape[0]):
        exemplesPerClasse[Y[i]] += 1

    # Calculem la mitjana de cadascún dels atributs
    mitjanaAtributs = numpy.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            mitjanaAtributs[j] += X[i][j]
    mitjanaAtributs = numpy.divide(mitjanaAtributs, X.shape[0])

    # Calculem la desviació típica
    desviacioTipicaAtributs = numpy.zeros(X.shape[1])
    acumulat = numpy.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            acumulat[j] += (X[i][j] - mitjanaAtributs[j]) ** 2
            pass
    desviacioTipicaAtributs = numpy.sqrt(acumulat / X.shape[0])

    return (exemplesPerClasse, mitjanaAtributs, desviacioTipicaAtributs)


# Normalitzem amb desviació típica 1
def normalitzaDades(X, mitjana):
    X_norm = X.copy()
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X_norm[i][j] = X[i][j] - mitjana[j]

    return (X_norm)

def calculaPCA(X, components):
    pca = sklearn.decomposition.PCA(n_components=components, copy=True, whiten=False,
                                    svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)

    return pca.fit(X).transform(X)

def calculaSVD(X):
    svd = sklearn.decomposition.TruncatedSVD()

    return svd.fit(X).transform(X)

def calculaLDA(X, Y):
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd')

    return lda.fit(X, Y).transform(X)

def displayScatterPlot(X, Y, name):
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.title('Scatter plot '+name)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.show()

def generaKNN(numVeins, X, Y):
    veins= KNeighborsClassifier(n_neighbors = numVeins)
    predictor = veins.fit(X, Y)
    return predictor


def compute_test(X, Y, classifyingFunction, cv):
    splitter = sklearn.model_selection.KFold(n_splits=cv)
    indices = splitter.split(X, Y)

    #TODO: Buscar también mejor dimensionalidad
    scoreActual = 0
    bestDimensiones = 0
    bestVecinos = 0
    for veins in range(1, 30):
        predictor = KNeighborsClassifier(n_neighbors=veins)
        for index_train, index_test in indices:
            X_train = X[index_train]
            X_test = X[index_test]
            Y_train = Y[index_train]
            Y_test = Y[index_test]
            Y_aux = [len(Y_test)]

            predictor.fit(X_train, Y_train)
            Y_aux = predictor.predict(X_test)

            score = accuracy_score(y_true=Y_test, y_pred=Y_aux, normalize=False)

            if score > scoreActual:
                print(str(veins) + " donen: " + str(score))
                scoreActual = score
                bestVecinos = veins

    return bestDimensiones, bestVecinos



if __name__ == "__main__":
    digits = sklearn.datasets.load_digits()
    X = digits.data
    Y = digits.target
    #print(X.shape, Y.shape)
    # Parte 1 del enunciado
    exemplesPerClasse, mitjana, desviacio = calculaEstadistiques(X, Y)
    #print(exemplesPerClasse, mitjana, desviacio)

    plt.imshow(numpy.reshape(X[Y == 3, :].mean(axis=0), [8, 8]))
    plt.show()

    # Parte dos del enunciado
    X_train, X_test, Y_train, Y_test = ns.train_test_split(X, Y, test_size=0.3)

    # Cogemos las medias y normalizamos
    aux, mitjana_train, aux2 = calculaEstadistiques(X_train, Y)
    aux, mitjana_test, aux2 = calculaEstadistiques(X_test, Y)
    X_train = normalitzaDades(X_train, mitjana_train)
    X_test = normalitzaDades(X_test, mitjana_test)

    print(X_train)
    print(X_test)

    # Parte 3
    # PCA
    X_pca = calculaPCA(X,2)
    # SVD
    #X_norm = normalitzaDades(X, mitjana)
    X_svd = calculaSVD(X) #Si pasamos la X normlaizada, nos saldrá igual que en PCA

    #Opcional: Visualización
    displayScatterPlot(X_pca, Y, "PCA")
    displayScatterPlot(X_svd, Y, "SVD")

    # LDA
    X_lda = calculaLDA(X, Y)
    displayScatterPlot(X_lda, Y, "LDA")

    # Parte 4
    n_dimensiones, n_vecinos = compute_test(X, Y, KNeighborsClassifier(n_neighbors = 1), 10)
    print("VECINOS: " + str(n_vecinos) + " DIMENSIONES: " + str(n_dimensiones))
    #predictor = generaKNN(n_vecinos)
    #Començem a predir!