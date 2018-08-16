import pandas as pd
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
import numpy as np

from perceptron1 import Perceptron


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker and color generators
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plot.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plot.xlim(xx1.min(), xx1.max())
    plot.ylim(xx2.min(), xx2.max())

    # plot class samples
    for index, cl in enumerate(np.unique(y)):
        plot.scatter(x=X[y == cl, 0],
                     y=X[y == cl, 1],
                     alpha=0.8,
                     c=colors[index],
                     marker=markers[index],
                     label=cl,
                     edgecolor='black')


def main():
    # Load in the iris dataset. Ths creates a DataForm object...
    iris_data_form = pd.read_csv('https://archive.ics.uci.edu/ml/'
                                 'machine-learning-databases/iris/iris.data',
                                 header=None)
    # Print last five lines to confirm data was loaded...
    print(iris_data_form.tail())

    """
    Extract the class labels from the first 100 samples.
    These will be our prediction targets for each sample.
    Once we have the class labels we convert them to
    a binary representation using the following convention:

    -1 = setosa
    +1 = versicolor

    """
    target_labels = iris_data_form.iloc[0:100, 4].values
    target_values = np.where(target_labels == 'Iris-setosa', -1, 1)

    """
    Extract the sepal length and petal length for each sample.
    These features will serve as our predictors.
    """
    predictor_features = iris_data_form.iloc[0:100, [0, 2]].values

    """
    For visualization purposes we plot the data.
    """
    plot.scatter(predictor_features[0:50, 0], predictor_features[0:50, 1],
                 color='red', marker='o', label='setosa')
    plot.scatter(predictor_features[50:100, 0], predictor_features[50:100, 1],
                 color='blue', marker='x', label='versicolor')
    plot.xlabel('sepal length [cm] ')
    plot.ylabel('petal length [cm] ')
    plot.legend(loc='upper left')
    plot.show()

    """
    Here is where we train our Perceptron (Linear Classification Model)
    """
    perceptron = Perceptron(learning_rate=0.1, n_length_iterator=10)
    perceptron.fit(predictor_features, target_values)

    # Below we create a graph showing the misclassification for each epoch
    plot.plot(range(1, len(perceptron.errors_) + 1),
              perceptron.errors_, marker='o')
    plot.xlabel('Epochs')
    plot.ylabel('Number of Updates')
    plot.show()

    # Run the trained model and plot a contour graph with the results
    plot_decision_regions(predictor_features, target_values, perceptron)
    plot.xlabel('sepal length [cm] ')
    plot.ylabel('petal length [cm] ')
    plot.legend(loc='upper left')
    plot.show()


if __name__ == "__main__":
    main()
