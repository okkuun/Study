# -*- coding: utf-8 -*-

import numpy as np
from common_func import Struct
from scipy.stats import multinomial, gamma, dirichlet, poisson
from scipy.special import digamma
from tqdm import tqdm
import matplotlib.pyplot as plt


class PoissonMixtureModel:
    def __init__(self) -> None:
        pass

    def fit(
            self, x: np.ndarray, k: int, iter: int = 200, method: str = 'gibbs'
    ) -> None:
        self._initialization(x, k)
        if method == 'gibbs':
            joint_func = self._joint_gibbs
        elif method == 'vb':
            joint_func = self._joint_vb
        else:
            Exception(f'You can set method as `gibbs`, `vb`, or `col-gibbs` but you set method = {method}')

        self.elbo_list = joint_func(iter)

    def _initialization(self, x: np.ndarray, k: int) -> None:
        self.x: np.ndarray = x.reshape(-1)
        self.k: int = k

        # initialize parameters
        self.s_dict = Struct(
            nu=np.random.random((len(self.x), k))
        )
        self.s_dict.e = self.s_dict.nu.copy()

        self.pi_dict = Struct(
            alpha=np.full(k, 1/k),
            alpha0=np.full(k, 1/k)
        )
        self.pi_dict.e = self.pi_dict.alpha.copy()
        self.pi_dict.log = digamma(self.pi_dict.alpha) - digamma(self.pi_dict.alpha.sum())
        # self.lam: np.ndarray = np.array([x.max() * i/k for i in range(1, k+1)])
        self.lam_dict = Struct(
            a=np.random.exponential(scale=1., size=k),
            b=np.random.random(size=k),
            a0=np.exp(np.full(k, 1.)),
            b0=np.full(k, 1.)
        )
        self.lam_dict.e = self.lam_dict.a / self.lam_dict.b
        self.lam_dict.log = digamma(self.lam_dict.a) - np.log(self.lam_dict.b+1e-8)

    def _joint_gibbs(self, iter: int) -> None:
        # burn in 期間をイテレーションの初めの20％とする
        burn_in: int = iter * 0.2
        thread = {'s': [], 'pi': [], 'lam': []}
        elbo_list = []

        for i in tqdm(range(iter)):

            # s
            self.s_dict.nu = np.exp(
                self.x[:, None] @ self.lam_dict.log[None] - self.lam_dict.e[None] + self.pi_dict.log[None]
            )
            self.s_dict.nu = self.s_dict.nu / self.s_dict.nu.sum(axis=1)[:, None]
            for _n in range(len(self.x)):
                self.s_dict.e[_n] = multinomial.rvs(n=1, p=self.s_dict.nu[_n])

            # pi
            self.pi_dict.alpha = self.s_dict.e.sum(axis=0) + self.pi_dict.alpha0
            self.pi_dict.e = dirichlet.rvs(self.pi_dict.alpha).reshape(-1)
            self.pi_dict.log = np.log(self.pi_dict.e+1e-8)

            # lam
            self.lam_dict.a = self.s_dict.e.T @ self.x + self.lam_dict.a0
            self.lam_dict.b = self.s_dict.e.sum(axis=0) + self.lam_dict.b0
            for _k in range(self.k):
                self.lam_dict.e[_k] = gamma.rvs(a=self.lam_dict.a[_k], scale=1/self.lam_dict.b[_k])
            self.lam_dict.log = np.log(self.lam_dict.e+1e-8)

            elbo_list.append(self.get_elbo())

            if i >= burn_in:
                thread['s'].append(self.s_dict.e.tolist())
                thread['pi'].append(self.pi_dict.e.tolist())
                thread['lam'].append(self.lam_dict.e.tolist())

        self.s_dict.e = np.array(thread['s']).mean(axis=0)
        self.pi_dict.e = np.array(thread['pi']).mean(axis=0)
        self.pi_dict.e /= self.pi_dict.e.sum()
        self.lam_dict.e = np.array(thread['lam']).mean(axis=0)

        return elbo_list

    def _joint_vb(self, iter: int) -> list:
        elbo_list = []
        for _ in tqdm(range(iter)):
            # s
            self.s_dict.nu = np.exp(
                self.x[:, None] @ self.lam_dict.log[None] - self.lam_dict.e[None] + self.pi_dict.log[None]
            )
            self.s_dict.nu = self.s_dict.nu / self.s_dict.nu.sum(axis=1)[:, None]
            self.s_dict.e = self.s_dict.nu

            # pi
            self.pi_dict.alpha = self.s_dict.e.sum(axis=0) + self.pi_dict.alpha0
            self.pi_dict.e = self.pi_dict.alpha / self.pi_dict.alpha.sum()
            self.pi_dict.log = digamma(self.pi_dict.alpha) - digamma(self.pi_dict.alpha.sum())

            # lam
            self.lam_dict.a = self.s_dict.e.T @ self.x + self.lam_dict.a0
            self.lam_dict.b = self.s_dict.e.sum(axis=0) + self.lam_dict.b0
            self.lam_dict.e = self.lam_dict.a / self.lam_dict.b
            self.lam_dict.log = digamma(self.lam_dict.a) - np.log(self.lam_dict.b+1e-8)

            elbo_list.append(self.get_elbo())

        return elbo_list

    # def _joint_colgibbs(self, iter: int) -> list:
    #     burn_in: int = iter * 0.2
    #     thread = {'s': []}
    #     elbo_list = []
    #     for i in tqdm(range(iter)):
    #     return elbo_list

    def get_elbo(self) -> float:
        # 参考: https://zhiyzuo.github.io/VI/#evidence-lower-bound-elbo
        E = (self.s_dict.e @ (self.x[:, None] @ self.lam_dict.log[None] - self.lam_dict.e[None]).T).sum()
        E += (self.s_dict.e @ self.pi_dict.log).sum()
        E += self.pi_dict.log.sum()
        E += self.lam_dict.log.sum()
        H = 0
        for _n in range(len(self.s_dict.e)):
            H += multinomial.entropy(n=1, p=self.s_dict.nu[_n]).item()
        for _k in range(self.k):
            H += gamma.entropy(a=self.lam_dict.a[_k], scale=1/self.lam_dict.b[_k]).item()
        H += dirichlet.entropy(alpha=self.pi_dict.alpha).item()

        return E - H

    def plot_elbo(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(self.elbo_list)), self.elbo_list)
        plt.show()
        plt.close()

    def get_params(self) -> dict:
        params = {}
        params['pi'] = self.pi_dict.e
        params['lam'] = self.lam_dict.e
        return params


def datagenerator(size: np.ndarray, lam: np.ndarray) -> np.ndarray:

    X = np.concatenate(
        [poisson.rvs(mu=lam[k], size=size[k]) for k in range(len(size))]
    )
    return X


if __name__ == '__main__':

    size = [100, 200]
    lam = [2, 8]

    data = datagenerator(size, lam)

    pmm = PoissonMixtureModel()

    pmm.fit(data, 2, method='gibbs')

    # 完全なelboではないから負になってない。変数に依存する項は含まれている。
    pmm.plot_elbo()

    params = pmm.get_params()
    x_range = np.arange(20)
    plt.figure()
    plt.hist(data, density=True, bins=15)
    plt.plot(x_range, params['pi'][0]*poisson.pmf(x_range, params['lam'][0]))
    plt.plot(x_range, params['pi'][1]*poisson.pmf(x_range, params['lam'][1]))
    plt.show()
    plt.close()
    print(params)
