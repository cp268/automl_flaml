import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from flaml.automl import AutoML
from pmlb import fetch_data


def main():
    df = fetch_data("adult")
    X = df.drop(["target"], axis=1)
    print(X.shape)
    y = df["target"]
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=21
    )
    automl = AutoML()
    automl.fit(X_train=X_train, y_train=y_train, task="classification")

    # retrieve best config and best learner
    print("Best ML leaner:", automl.best_estimator)
    print("Best hyperparmeter config:", automl.best_config)

    # get predictions
    preds = automl.predict(X_test)

    # print evaluation scores
    print(classification_report(y_test, preds))

    # Visualize feature importance
    plt.barh(
        automl.model.estimator.feature_name_,
        automl.model.estimator.feature_importances_,
    )


if __name__ == "__main__":
    main()
