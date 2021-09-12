# -*- coding: utf-8 -*-

import numpy as np
from common_func import Struct


class BayesianLinearRegression:
    def __init__(self) -> None:
        pass

    def fit(
        self, X: np.ndarray, y: np.ndarray,
        lam_y: np.ndarray = None, Lam_w: np.ndarray = None
    ) -> None:
        self.y = y
        self.X = X
        num_samples, num_dims = X.shape
        if lam_y is None:
            self.lam_y = np.array([1.])
        elif len(self.lam_y) != 1:
            Exception("The shape of lam_y must be (1,)")
        else:
            self.lam_y = lam_y

        if Lam_w is None:
            Lam_w = np.identity(n=num_dims)
        elif Lam_w.shape != (num_dims, num_dims):
            Exception("The shape of Lam_w must be (w's dims, w's dims)")
        self.w_dist = Struct(
            mu=np.zeros(shape=num_dims),
            Lam=Lam_w.copy()
        )

        # posterior distribution of w
        self.w_dist.Lam = self.lam_y * self.X.T @ self.X + self.w_dist.Lam
        _mu = self.lam_y * self.X.T @ self.y + Lam_w @ self.w_dist.mu
        self.w_dist.mu = np.linalg.inv(self.w_dist.Lam) @ _mu

    def predict_expectation(self, X_test: np.ndarray) -> np.ndarray:
        new_y_dist = self.predict_distribution(X_test)
        return new_y_dist.mu

    def predict_distribution(self, X_test: np.ndarray) -> dict:
        new_y_dist = Struct()
        new_y_dist.lam = 1 / (
            1/self.lam_y + np.sum(X_test @ np.linalg.inv(self.w_dist.Lam) @ X_test.T)
        )
        new_y_dist.mu = X_test @ self.w_dist.mu
        return new_y_dist

    def get_log_modelevidence(self):

        pass


# def generator(seed=1003, num_dims=2, per=20):
#     x_range = (0, 10)
#     pass

def y_generator(X):
    w = np.array([0.5, 2.4, -3.1, 1.3])
    np.random.seed(1003)
    y = X @ w + np.random.normal(size=len(X))
    return y


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # generate data

    size = 200
    x_axis = np.linspace(0, 10, num=size)
    test_idx = np.random.binomial(n=1, p=0.2, size=size).astype(bool)

    x_train, x_test = x_axis[~test_idx], x_axis[test_idx]
    x_train_sta = (x_train - x_train.mean()) / x_train.std()
    x_test_sta = (x_test - x_train.mean()) / x_train.std()

    X_train = np.concatenate([
        np.ones_like(x_train_sta)[:, None],
        x_train_sta[:, None],
        x_train_sta[:, None]**2,
        x_train_sta[:, None]**3
    ], axis=1)
    X_test = np.concatenate([
        np.ones_like(x_test_sta)[:, None],
        x_test_sta[:, None],
        x_test_sta[:, None]**2,
        x_test_sta[:, None]**3
    ], axis=1)

    # サンプルデータ作成（これ意味あるんだろうか）
    y_train = y_generator(X_train)
    y_test = y_generator(X_test)

    # fit & predict
    blr = BayesianLinearRegression()
    blr.fit(X_train, y_train)
    y_pred_dist = blr.predict_distribution(X_test)

    fig, ax = plt.subplots()
    ax.plot(x_test, y_pred_dist.mu, c='orange', label='Expectation')
    ax.fill_between(
        x_test,
        y_pred_dist.mu + 1 / np.sqrt(y_pred_dist.lam),
        y_pred_dist.mu - 1 / np.sqrt(y_pred_dist.lam),
        color="orange", alpha=0.4
    )
    ax.scatter(x_test, y_test, color='black', label='True')
    plt.legend()
    plt.savefig('./3-3_result.png')
    plt.show()
    plt.close()
