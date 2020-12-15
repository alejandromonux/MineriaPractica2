import numpy
import sklearn
import sklearn.datasets
import sklearn.model_selection as ns
import sklearn.discriminant_analysis
import sklearn.decomposition
import sklearn.neighbors
import sklearn.metrics
import matplotlib.pyplot as plt
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
    #figure = plt.figure()
    #axis = figure.add_subplot(1, 1, 1)
    #colors = ("red","green","yellow","black","pink","purple","orange", "brown", "coral", "lime")

    #i=0
    #for x in X:
    #    axis.scatter(x[0], x[1], alpha=0.8, c=colors[Y[i]],edgecolors='none', s=30, label=Y[i])
    #    i+=1
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

#                                                   Numero de folds
def compute_test(X,    Y,    classifyingFunction,   cv):
    splitter = sklearn.model_selection.KFold(n_splits=cv)
    indices = splitter.split(X, Y)  # Dentro de indices existirán diez grupos de dos arrays, uno de estos dos con índices de training (90%)y otro de test(10%) que se irán intercalando

    #Tenemos que minimizar la classifying function??

    scoreActual=0
    bestDimensiones=0
    bestVecinos=0
    train_test = []
    #for index_train, index_test in indices:
    #    scores.append(0)
    for index_train, index_test in indices:
        #Comprovaremos para todas las dimensionalidades
        for dimensiones in range(1,64):
            #Cambiamos la dimensionalidad
            new_X = calculaPCA(X, dimensiones)
            #Cambiar new_X para que sea sólo los índices de train. Y otra para los de test. Hacer el cross_val_score con test.
            #TODO: Que esta dimensión coincida con los índices
            #X_train, X_test, Y_train, Y_test = ns.train_test_split(new_X, Y, test_size=(1/cv))
            X_train = new_X[index_train];
            #X_test = new_X[index_test];
            Y_train = Y[index_train];
            #Y_test = Y[index_test];

            # Miramos los vecinos, el rango es un numero arbitrario, de momento el número de ejemplos de train
            #classifyingFunction.fit(X_train, Y)
            for veins in range(1, 180):
                ##Probar numero de vecinos con esta dimensionalidad
                #Opción 1: Con SVC: cross_val_score(estimator=classifyingFunction, X=X_test, y=Y, cv=cv)
                #Opción 2: Con Kneighbour s
                predictor = KNeighborsClassifier(n_neighbors = veins)
                scores = cross_val_score(estimator=predictor, X=X_train, y=Y_train, cv=cv)
                #Calcul de la mitjana de scores
                score = sum(scores)/len(scores)
                #Después de hacer cosas, si estamos en una combinación con una score mejor que la anterior, cogeremos esta.
                if  score > scoreActual:
                    bestDimensiones = dimensiones
                    bestVecinos = veins
                    if len(train_test) == 2:
                        train_test[0] = index_train
                        train_test[1] = index_test
                    else:
                        train_test.append(index_train)
                        train_test.append(index_test)



    return bestDimensiones, bestVecinos, train_test


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
    n_vecinos, n_dimensiones = compute_test(X, Y, SVC(gamma='auto', decision_function_shape='ovo'), 10)
    print("VECINOS: " + n_vecinos + " DIMENSIONES: " + n_dimensiones)
    #predictor = generaKNN(n_vecinos)
    #Començem a predir!