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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


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
    acumulat = numpy.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            acumulat[j] += (X[i][j] - mitjanaAtributs[j]) ** 2
            pass
    desviacioTipicaAtributs = numpy.sqrt(acumulat / X.shape[0])

    return exemplesPerClasse, mitjanaAtributs, desviacioTipicaAtributs


# Normalitzem amb desviació típica 1
def normalitzaDades(X, mitjana):
    X_norm = X.copy()
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X_norm[i][j] = X[i][j] - mitjana[j]

    return X_norm


def calculaPCA(X, components):
    pca = sklearn.decomposition.PCA(n_components=components, copy=True, whiten=False, svd_solver='auto', tol=0.0,
                                    iterated_power='auto', random_state=None)
    return pca.fit(X).transform(X)


def calculaSVD(X):
    svd = sklearn.decomposition.TruncatedSVD()
    return svd.fit(X).transform(X)


def calculaLDA(X, Y):
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd')
    return lda.fit(X, Y).transform(X)


def displayScatterPlot(X, color, name):
    plt.scatter(X[:, 0], X[:, 1], c=color)
    plt.title('Scatter plot ' + name)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar()
    plt.show()


def generaKNN(numVeins, X, Y):
    veins = KNeighborsClassifier(n_neighbors=numVeins)
    predictor = veins.fit(X, Y)
    return predictor


def compute_test(X, Y, classifyingFunction, cv):
    splitter = sklearn.model_selection.KFold(n_splits=cv)
    indices = splitter.split(X, Y)
    X_new = []
    for dimensio in range(1, 64):
        X_new.append(calculaPCA(X, dimensio))

    scoreActual = 0
    bestDimensiones = 0
    bestVecinos = 0
    progress = 1
    for index_train, index_test in indices:
        print("Progress (" + str(progress) + "/10)")
        progress += 1
        for veins in range(1, 50):
            predictor = KNeighborsClassifier(n_neighbors=veins, n_jobs=-1)
            for dimensio in range(1, 64):
                X_train = X_new[dimensio - 1][index_train]
                X_test = X_new[dimensio - 1][index_test]
                Y_train = Y[index_train]
                Y_test = Y[index_test]

                # Entrena el modelo con estos valores
                predictor.fit(X_train, Y_train)
                # Testeamos el modelo con los datos de test
                Y_aux = predictor.predict(X_test)
                # Compara con Y_aux cuantos valores se han predecido correctamente con el conjunto Y_test
                score = accuracy_score(y_true=Y_test, y_pred=Y_aux, normalize=False)

                if score > scoreActual:
                    print("Accepted: " + str(veins) + " vecinos y " + str(dimensio) + " dimensiones. Score: " + str(
                        score))
                    scoreActual = score
                    bestVecinos = veins
                    bestDimensiones = dimensio

    return bestDimensiones, bestVecinos


def cercaDeParametres(cv, X, Y):
    n_neighbours_max = 64
    n_dimensions_max = 64
    n_neighbours = range(1, n_neighbours_max)
    n_dimensions = range(1, n_dimensions_max)
    clf = GridSearchCV(estimator=KNeighborsClassifier(), n_jobs=-1, param_grid={'n_neighbors': n_neighbours}, cv=cv)

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


if __name__ == "__main__":
    digits = sklearn.datasets.load_digits()
    X = digits.data
    Y = digits.target
    # print(X.shape, Y.shape)

    # ---> Parte 1 del enunciado
    exemplesPerClasse, mitjana, desviacio = calculaEstadistiques(X, Y)
    # print(exemplesPerClasse, mitjana, desviacio)

    # Cogemos todas las filas que sean de la clase Y (indicada), y hacemos la media de todas la filas para quedarnos con
    # un array de 64 posiciones (media de atributes) el cual hacemos un reshape para obtener una matrix 8x8
    plt.imshow(numpy.reshape(X[Y == 3, :].mean(axis=0), [8, 8]))
    plt.show()

    # ---> Parte 2 del enunciado
    # Con la siguiente función obtenemos diferentes sets de entrenamiento i test, siendo el 70% train i 30% el conjunto
    # de tests.
    X_train, X_test, Y_train, Y_test = ns.train_test_split(X, Y, test_size=0.3)

    # Cogemos las medias y normalizamos
    # En este caso solo nos interesa la media de X_traing y X_test, los valores aux, aux2 e Y, es para que no haya
    # problemas de sintaxis
    aux, mitjana_train, aux2 = calculaEstadistiques(X_train, Y)
    aux, mitjana_test, aux2 = calculaEstadistiques(X_test, Y)
    X_train = normalitzaDades(X_train, mitjana_train)
    X_test = normalitzaDades(X_test, mitjana_test)

    # Conjunts train i test normalitzats
    print(X_train)
    print(X_test)

    # ---> Parte 3 del enunciado
    # Como se puede observar de cada transformación se aplica un fit, para ajustar la técnica al modelo de datos, y un
    # transform para aplicar en esos datos lo que pretendemos hacer

    # PCA
    # Mayor dispersión y reducción de la dimensionalidad.
    # Cambiar las dimensiones manteniendo la máxima varianza posible.
    X_pca = calculaPCA(X, 2)
    # SVD
    X_svd = calculaSVD(X)  # Si pasamos la X normlaizada, nos saldrá igual que en PCA

    # Opcional: Visualización de los datos
    displayScatterPlot(X_pca, Y, "PCA")
    displayScatterPlot(X_svd, Y, "SVD")

    # LDA. Otro tipo de representación
    # Aumentar la visualización de los datos.
    X_lda = calculaLDA(X, Y)
    displayScatterPlot(X_lda, Y, "LDA")

    # ---> Parte 4 del enunciado
    # Encontramos el mejor conjunto de veciones y dimensiones mediante KNN
    n_dimensiones, n_vecinos = compute_test(X, Y, KNeighborsClassifier(n_neighbors=1), 10)
    print("Mejor combinación --> VECINOS: " + str(n_vecinos) + " DIMENSIONES: " + str(n_dimensiones))

    # X_new = calculaPCA(X, n_dimensiones)
    # predictor = generaKNN(n_vecinos, X_new, Y)

    # Finalmente mostramos una gráfica donde se muestra los diferentes resultados de predicción según vecinos y
    # dimensiones, mediante GridSearchCV
    cercaDeParametres(10, X, Y)
