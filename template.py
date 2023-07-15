import numpy as np
import csv
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# %%

TEST_SIZE = 0.3
K = 3


class NN:

    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, testing_features, k):
        predictions = []
        for feature in testing_features:
            distances = np.sqrt(np.sum(np.square(np.subtract(self.trainingFeatures, feature)), axis=1))
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = [self.trainingLabels[i] for i in nearest_indices]
            prediction = np.argmax(np.bincount(nearest_labels))
            predictions.append(prediction)
        return predictions

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

    return df.iloc[:, :-1].values.tolist(), df.iloc[:, -1].values.tolist()  # features, labels


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


def confusion_matrix_hand_made(labels, predictions):
    tp = np.sum(np.logical_and(labels, predictions))  # true positive
    fp = np.sum(np.logical_and(np.logical_not(labels), predictions))  # false positive
    tn = np.sum(np.logical_and(np.logical_not(labels), np.logical_not(predictions)))
    # true negative
    fn = np.sum(np.logical_and(labels, np.logical_not(predictions)))  # false negative
    return tp, fp, tn, fn


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1). -we added confusion matrix-

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    tp, fp, tn, fn = confusion_matrix_hand_made(labels, predictions)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1


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
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)
    conufusion_mtrx = confusion_matrix_hand_made(y_test, predictions)


    # Print results
    print("**** k = 3 Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Confusion Matrix: \n")
    print(conufusion_mtrx[0], conufusion_mtrx[1])
    print(conufusion_mtrx[2], conufusion_mtrx[3])

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)
    conufusion_mtrx = confusion_matrix_hand_made(y_test, predictions)


    # Print results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("Confusion Matrix: \n")
    print(conufusion_mtrx[0], conufusion_mtrx[1])
    print(conufusion_mtrx[2], conufusion_mtrx[3])


if __name__ == "__main__":
    main()
