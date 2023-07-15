import numpy as np
import csv
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.neural_network import MLPClassifier

# %%
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.3
K = 3


class NN:

    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, testing_features, k):
        classifier = KNeighborsClassifier(n_neighbors=k, p=2)  # p=2 is euclidean distance
        classifier.fit(self.trainingFeatures, self.trainingLabels)
        label_prediction = classifier.predict(testing_features)
        return label_prediction

        """
        Given a list of features vectors of testing examples
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors
        """


# %%


def load_data(filename):
    df = pd.read_csv(filename)  # data frame
    """
    Load spam data from a CSV file `filename` and convert into a list of
    features vectors and a list of target labels. Return a tuple (features, labels).

    features vectors should be a list of lists, where each list contains the
    57 features vectors

    labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """

    return df.iloc[:, :-1].values, df.iloc[:, -1].values  # features, labels


# %%


def preprocess(features):
    """
    normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation
    """
    features = np.array(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean) / std
    return features.tolist()


def train_mlp_model(features, labels):
    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """
    model = MLPClassifier(random_state=1, hidden_layer_sizes=(10, 5), max_iter=2000, activation='logistic')
    model.fit(features, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1). -we added confusion matrix-

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    confusion_mtrx = confusion_matrix(labels, predictions)
    return accuracy, precision, recall, f1, confusion_mtrx


# %%


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(sys.argv[1])
    # features = preprocess(features)  # MLP

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1, confusion_mtrx = evaluate(y_test, predictions)

    # Print results
    print("**** k = 3 Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Confusion Matrix: \n", confusion_mtrx)

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1, confusion_mtrx = evaluate(y_test, predictions)

    # Print results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Confusion Matrix:\n ", confusion_mtrx)


if __name__ == "__main__":
    main()
