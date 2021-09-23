# -*- coding: utf-8 -*-

from code_green.common_func import Struct
import numpy as np
# from common_func import Struct
from scipy.stats import poisson, multinomial, gamma
from tqdm import tqdm


class PoissonMixtureModel:
    def __init__(self) -> None:
        pass

    def fit(
            self, x: np.ndarray, k: int, iter: int = 20, method: str = 'gibbs'
    ) -> None:
        self.x: np.ndarray = x.reshape(-1)
        self.k: int = k
        self.s: np.ndarray = np.empty((len(self.x), len(self.k)))
        self.method: str = method

        # initialize parameters
        self.pi: np.ndarray = np.full(k, 1/k)
        self.lam: np.ndarray = np.array([x.max() * i/k for i in range(1, k+1)])
        self.lam_dict = Struct(
            a = np.random.exponential(scale=1., size=k),
            b = np.random.exponential
        )


    def _fit_gibbs(self, iter: int) -> None:
        for i in tqdm(range(iter)):
            nu = np.exp(
                self.x[:, None] @ np.log(self.lam+1e-8)[None] - self.lam[None] + np.log(self.pi+1e-8)[None]
            )
            nu = nu / nu.sum(axis=1)[None]
            for _n in range(len(self.x)):
                self.s[_n] = multinomial.rvs(n=1, p=nu[_n])

            for _k in range(self.k):

