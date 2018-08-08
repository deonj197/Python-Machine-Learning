import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

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

if __name__ == "__main__":
    main()
