# Email_Spam_Filter

## About this project
This project focuses on the development and evaluation of two models, namely K-Nearest Neighbors (K-NN) and Multilayer Perceptron (MLP), for email spam detection. The models were trained and tested using the provided spambase.csv file, which consists of 4601 examples represented by 58 numbers. The final number in each row indicates whether the email was classified as spam (1) or not (0). The evaluation of the models includes measuring accuracy, precision, recall, F1-score, and Confusion matrix on the test set. To understand the attributes' meaning, please refer to the provided [link](https://archive.ics.uci.edu/dataset/94/spambase) for further details.

## Setup

To install the project, run the following command in the terminal:

```bash
git clone https://github.com/faris771/Email_Spam_Filter.git
cd Email_Spam_Filter
```

To install the required packages, run the following command in the terminal:

```bash
pip install -r requirements.txt
```

To run the project, run the following command in the terminal:

```bash
python main.py spambase.csv
```

## Results
KNN works by finding the k nearest neighbours of a given example and then using the labels of those neighbours to predict the label of the example. MLP, on the other hand, works by creating a model that learns the relationship between the features of an example and its label.
In the case of spam filtering, KNN is often better than MLP because it is less sensitive to noise. MLP can sometimes be fooled by spam that is well-written or that contains images. KNN is also less computationally expensive than MLP, which can be important for large datasets.
However, MLP can sometimes be more accurate than KNN, especially for complex tasks. MLP can also learn to recognize patterns that KNN cannot, such as the use of certain words or phrases that are often associated with spam.
Ultimately, the best algorithm for spam filtering depends on the specific dataset and the desired level of accuracy. For our dataset the MLP had on average better results in comparison to the KNN algorithm.
We experimented by calculating the confusion matrix, accuracy, precision, recall, and f1 score manually rather than using the scikit library for authenticity.



![mtrxxx](https://github.com/faris771/Email_Spam_Filter/assets/70337488/b6b82bf0-90b5-476b-b9e2-e87d719a3310)

## Contributors
* [Faris Abufarha](https://github.com/faris771)
* [Rasheed AL Qubbaj](https://github.com/Rasheed-Al-Qobbaj)
 

