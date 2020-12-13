import numpy
import sklearn.model_selection as ms
import sklearn.decomposition as dec
import sklearn.discriminant_analysis as da
import sklearn.neighbors as n
import sklearn.datasets
import matplotlib.pyplot as plt


def calcEstBas(X, Y):
    mitj = numpy.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            mitj[j] += X[i][j]
    mitj = mitj / X.shape[0]

    desv = numpy.zeros(X.shape[1])
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            desv[j] += (X[i][j] - mitj[j]) ** 2
    desv = numpy.sqrt(desv / X.shape[0])

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

def compute_test(x_test, y_test, clf, cv):
    Kfolds = sklearn.model_selection.Kfold(...)

    scores = []
    for i, j in Kfolds:
        # ...
        scores.append()
    return scores


if __name__ == '__main__':
    #   1
    digits = sklearn.datasets.load_digits()
    X = digits.data  # info 8x8 matrix of each example
    Y = digits.target  # info of type of example

    print(X.shape)
    print(Y.shape)
    # print(digits.DESCR)

    # Calculation of Basci estadistics
    mitjana, desviacio_tipica, n_x_classe = calcEstBas(X, Y)
    print('Mean:')
    print(mitjana)
    print('Deviation:')
    print(desviacio_tipica)
    print('Number of examples in class:')
    print(n_x_classe)

    # Plot mean of one class of examples
    plt.imshow(numpy.reshape(X[Y == 5, :].mean(axis=0), [8, 8]))
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
    pca = dec.PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print('PCA 2 Components:')
    print(X_pca)
    # Plot PCA
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y)
    plt.title('PCA')
    plt.colorbar()
    plt.show()

    # 2 Component SVD
    svd = dec.TruncatedSVD()
    X_svd = svd.fit_transform(X)
    print('SVD 2 Components:')
    print(X_svd)
    # Plot SVD
    plt.scatter(X_svd[:, 0], X_svd[:, 1], c=Y)
    plt.title('Truncated SVD')
    plt.colorbar()
    plt.show()

    # Discrimination of data LDA
    lda = da.LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X, Y)
    print('LDA 2 Components:')
    print(X_lda)
    # Plot LDA
    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=Y)
    plt.title('LDA')
    plt.colorbar()
    plt.show()

    #   4
    # 10 fold cross validation
    kf = ms.KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)

    # Test function
    # compute_test()

    # K-NN
    knn = n.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, Y)
