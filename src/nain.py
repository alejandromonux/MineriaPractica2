import numpy
import sklearn.model_selection as ms
import sklearn.decomposition as dec
import sklearn.discriminant_analysis as da
import sklearn.metrics as met
import sklearn.neighbors as n
import sklearn.datasets
import matplotlib.pyplot as plt


def calcEstBas(X, Y):
    # Mean of each attribute
    mitj = numpy.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            mitj[j] += X[i][j]
    mitj = mitj / X.shape[0]

    # Tipical deviation
    desv = numpy.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            desv[j] += (X[i][j] - mitj[j]) ** 2
    desv = numpy.sqrt(desv / X.shape[0])

    # N examples per class
    n_x_c = numpy.zeros(10)
    for t in Y:
        n_x_c[t] += 1

    return mitj, desv, n_x_c


def normalizeZScore(X):
    for i in range(0, X.shape[0]):
        XMean = X[i].mean(axis=0)
        for j in range(0, X.shape[1]):
            X[i][j] -= XMean
    return X


def compute_test(X, Y, cv):
    # 10 fold cross validation
    # kf = ms.KFold(n_splits=10)
    # for train_index, test_index in kf.split(X):
    #    print("TRAIN:", train_index, "TEST:", test_index)

    kf = ms.KFold(n_splits=cv)
    indices = kf.split(X, Y)

    scoreActual = 0
    best_dimensions = 0
    best_neighbors = 0
    for index_train, index_test in indices:
        for neighbors in range(1, 180):
            predictor = n.KNeighborsClassifier(n_neighbors=neighbors)
            for dimension in range(1, 64):
                X_new = calculate_PCA(X, dimension)

                X_train = X_new[index_train]
                X_test = X_new[index_test]
                Y_train = Y[index_train]
                Y_test = Y[index_test]
                Y_aux = [len(Y_test)]

                predictor.fit(X_train, Y_train)
                Y_aux = predictor.predict(X_test)

                score = met.accuracy_score(y_true=Y_test, y_pred=Y_aux, normalize=False)

                if score > scoreActual:
                    scoreActual = score
                    best_neighbors = neighbors
                    best_dimensions = dimension

    return best_dimensions, best_neighbors


def cercaDeParametres(cv, X, Y, size):
    n_neighbours_max = size
    n_dimensions_max = size
    n_neighbours = range(1, n_neighbours_max)
    n_dimensions = range(1, n_dimensions_max)
    clf = ms.GridSearchCV(estimator=n.KNeighborsClassifier(),
                          n_jobs=-1,
                          param_grid={'n_neighbors': n_neighbours}
                          , cv=cv)

    score = []
    for dimensions in n_dimensions:
        X_new = calculate_PCA(X, dimensions)
        clf.fit(X_new, Y)
        score.append(clf.cv_results_['mean_test_score'])

    # displayScatterPlot(n_dimensions, n_neighbours, score,"GridCV")

    for i in n_neighbours:
        x = [i] * (n_dimensions_max - 1)
        plt.scatter(n_neighbours, x, c=score[i - 1])
    plt.title('Scatter plot GridCV')
    plt.xlabel('Neighbours')
    plt.ylabel('Dimensions')
    plt.colorbar()
    plt.show()


def calculate_PCA(X, n_c):
    pca = dec.PCA(n_components=n_c)
    return pca.fit(X).transform(X)


def calculate_SVD(X):
    svd = dec.TruncatedSVD()
    return svd.fit(X).transform(X)


def calculate_LDA(X, Y):
    lda = da.LinearDiscriminantAnalysis()
    return lda.fit(X, Y).transform(X)


if __name__ == '__main__':
    #   1
    digits = sklearn.datasets.load_digits()
    X = digits.data  # info 8x8 matrix of each example
    Y = digits.target  # info of type of example

    print(X.shape)
    print(Y.shape)
    # print(digits.DESCR)

    # Calculation of Basic estadistics
    mitjana, desviacio_tipica, n_x_classe = calcEstBas(X, Y)
    print('Mean:')
    print(mitjana)
    print('Deviation:')
    print(desviacio_tipica)
    print('Number of examples in class:')
    print(n_x_classe)

    # Plot mean of one class of examples
    plt.imshow(numpy.reshape(X[Y == 6, :].mean(axis=0), [8, 8]))
    plt.show()

    #   2
    # Division data in train and test
    X_train, X_test, Y_train, Y_test = ms.train_test_split(X, Y, test_size=0.3)
    # print(X_train)
    # print(X_test)
    # print(Y_train)
    # print(Y_test)

    # Normalized data centered in 0 and 1 tipic deviation
    XN_train = normalizeZScore(X_train)
    XN_test = normalizeZScore(X_test)
    print('Normalized Train:')
    print(XN_train)
    print('Normalized Test:')
    print(XN_test)

    #   3
    # 2 Component PCA
    X_pca = calculate_PCA(X, 2)
    print('PCA 2 Components:')
    print(X_pca)
    # Plot PCA
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y)
    plt.title('PCA')
    plt.colorbar()
    plt.show()

    # 2 Component SVD
    X_svd = calculate_SVD(X)
    print('SVD 2 Components:')
    print(X_svd)
    # Plot SVD
    plt.scatter(X_svd[:, 0], X_svd[:, 1], c=Y)
    plt.title('Truncated SVD')
    plt.colorbar()
    plt.show()

    # Discrimination of data LDA
    X_lda = calculate_LDA(X, Y)
    print('LDA 2 Components:')
    print(X_lda)
    # Plot LDA
    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=Y)
    plt.title('LDA')
    plt.colorbar()
    plt.show()

    #   4
    # Test function
    # Find the best pair of number of neighbors and dimension in KNN
    n_dimensions, n_neighbors = compute_test(X, Y, 10)
    print("Best = N_Neighbors= " + str(n_neighbors) + " & N_Dimensions= " + str(n_dimensions))
    # Best = N_Neighbors= 3 & N_Dimensions= 27
    # K-NN
    X_new = calculate_PCA(X, n_dimensions)
    knn = n.KNeighborsClassifier(n_neighbors=n_neighbors)
    predict = knn.fit(X_new, Y)
    # Plot of the diferent prediction results depending on dimension and neighbors with GridSearchCV
    cercaDeParametres(10, X, Y, 64)
