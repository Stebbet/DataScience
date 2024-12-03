import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import numpy as np
from scipy import stats


def clean_data(filename):
    """
    Read the csv file and clean the data
    :param filename: filename
    :return: The dataset
    """
    data = pd.read_csv(filename)
    data.drop('id', axis='columns', inplace=True)

    data = data.dropna()
    # Use Z-score to clean the data from outliers further than 3 Standard Deviations
    z = np.abs(stats.zscore(data))
    data = data[(z < 3).all(axis=1)]

    return data

def feature_selection(data):


    X = df[:-1]
    y = df[-1]

    info = SelectKBest(mutual_info_classif, k='all')
    features = info.fit(X, y)
    df_scores = pd.DataFrame(features.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    series = pd.Series(features.scores_, index=X.columns)
    series.nlargest(10).plot.barh()
    plt.title("Mutual Information using SelectKBest")
    plt.tight_layout()
    plt.show()

    model = DecisionTreeClassifier()
    model.fit(X, y)

    (pd.Series(model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh'))
    plt.title("Gini Importance of Features")
    plt.tight_layout()
    plt.show()


def prune(X, y):
    """
    Prune the decision tree classifier, this will help improve the performance of the model as well as its fitting speed
    :param X: data
    :param y: labels
    :return: None, execting plots of different alpha and their training and testing scores
    """
    x_train, x_test, y_train, y_test = train_test_split(X, y,stratify=y)

    clf = tree.DecisionTreeClassifier(random_state=0)

    # Post-Pruning
    path = clf.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas[:-1], path.impurities[:-1]

    plt.figure(figsize=(6, 6))
    plt.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    plt.xlabel("effective alpha")
    plt.ylabel("total impurity of leaves")
    plt.title("Total Impurity vs effective alpha for training set")
    plt.show()

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.plot()

    train_scores = [clf.score(x_train, y_train) for clf in clfs]
    test_scores = [clf.score(x_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.show()

if __name__ == '__main__':
    # Compares the accuracy of the unpruned and the pruned decision tree

    df = clean_data('internet_service_churn.csv')


    # Pre-pruning by using the attributes with the most information gain
    # feature_selection(df)
    X = df[["subscription_age", "bill_avg", "remaining_contract","download_avg", "upload_avg"]]
    y = df[['churn']]

    x_train, x_test, y_train, y_test = train_test_split(X, y,stratify=y)

    # Get the alpha from pruning the tree
    # prune(X, y)
    alpha = 0.00015

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

    clf2 = DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    clf2.fit(x_train, y_train)
    p_y_train_pred = clf2.predict(x_train)
    p_y_test_pred = clf2.predict(x_test)

    train = confusion_matrix(y_train, y_train_pred)
    test = confusion_matrix(y_test, y_test_pred)
    _train = confusion_matrix(y_train, p_y_train_pred)
    _test = confusion_matrix(y_test, p_y_test_pred)

    print("accuracy of un-pruned tree using confusion matrix for train data =",
          (train[0, 0] + train[1, 1]) * 100 / (train[0, 0] + train[1, 1] + train[0, 1] + train[1, 0]))

    print("accuracy of un-pruned tree using confusion matrix for test data =",
          (test[0, 0] + test[1, 1]) * 100 / (test[0, 0] + test[1, 1] + test[0, 1] + test[1, 0]))

    print("accuracy of pruned tree using confusion matrix for train data =",
          (_train[0, 0] + _train[1, 1]) * 100 / (_train[0, 0] + _train[1, 1] + _train[0, 1] + _train[1, 0]))

    print("accuracy of pruned tree using confusion matrix for test data =",
          (_test[0, 0] + _test[1, 1]) * 100 / (_test[0, 0] + _test[1, 1] + _test[0, 1] + _test[1, 0]))


