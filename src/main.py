import numpy
import sklearn
import sklearn.datasets
import sklearn.model_selection as ns
import sklearn.decomposition
import sklearn.neighbors
import sklearn.metrics
import matplotlib.pyplot as plt

def calculaEstadistiques(X, Y):

    #Busquem el nombre de elements de entrenament per classe
    exemplesPerClasse = numpy.zeros(10)
    for i in range(0, Y.shape[0]):
        exemplesPerClasse[Y[i]] += 1

    #Calculem la mitjana de cadascún dels atributs
    mitjanaAtributs = numpy.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            mitjanaAtributs[j] += X[i][j]
    mitjanaAtributs = numpy.divide(mitjanaAtributs, X.shape[0])


    #Calculem la desviació típica
    desviacioTipicaAtributs = numpy.zeros(X.shape[1])
    acumulat = numpy.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            acumulat[j] += (X[i][j]-mitjanaAtributs[j])**2
            pass
    desviacioTipicaAtributs = numpy.sqrt(acumulat/X.shape[0])


    return (exemplesPerClasse, mitjanaAtributs, desviacioTipicaAtributs)

#Normalitzem amb desviació típica 1
def normalitzaDades(X_train, X_test, mitjana):
    for i in range(0, X_train.shape[0]):
        for j in range(0, X_train.shape[1]):
            X_train[i][j] -= mitjana[j]

    for i in range(0, X_test.shape[0]):
        for j in range(0, X_test.shape[1]):
            X_test[i][j] -= mitjana[j]

    return(X_train, X_test)

if __name__ == "__main__":
    digits = sklearn.datasets.load_digits()
    X = digits.data
    Y = digits.target
    print(X.shape, Y.shape)
    #Parte 1 del enunciado
    exemplesPerClasse, mitjana, desviacio = calculaEstadistiques(X, Y)
    print(exemplesPerClasse, mitjana, desviacio)

    plt.imshow(numpy.reshape(X[Y == 6, :].mean(axis=0), [8,8]))
    plt.show()

    #Parte dos del enunciado
    X_train, X_test, Y_train,Y_test = ns.train_test_split(X, Y, test_size=0.3)

    #Cogemos las medias y normalizamos 
    aux, mitjana_train, aux2 = calculaEstadistiques(X_train, Y)
    aux, mitjana_test, aux2 = calculaEstadistiques(X_test, Y)
    X_train, X_test = normalitzaDades(X_train, X_test, mitjana_train, mitjana_test)