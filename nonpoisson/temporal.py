from nonpoisson import paths
from nonpoisson import catalogs
import statsmodels.discrete.discrete_model
from statsmodels.distributions.empirical_distribution import ECDF

import numpy as np
import random
import seaborn as sns
from scipy.signal import medfilt
import scipy.stats as st
import pickle
from scipy.special import gammainc, betainc
from datetime import datetime as dt
from matplotlib.lines import Line2D
import statistics
import scipy.interpolate as si
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

sns.set_style("darkgrid", {"axes.facecolor": ".9", 'font.family': 'Ubuntu'})


class CatalogVariability(object):

    def __init__(self):

        self.n = []
        self.n2 = []
        self.ratio = []
        self.dt = []
        self.t1 = []
        self.t2 = []
        self.ti1 = []
        self.ti2 = []
        self.stats = {}

    def add(self, n, n2=None, ratio=None, dt=None, t1=None, t2=None, ti1=None, ti2=None):

        self.n.append(n)
        self.n2.append(n2)
        self.ratio.append(ratio)
        self.dt.append(dt)
        self.t1.append(t1)
        self.t2.append(t2)
        self.ti1.append(ti1)
        self.ti2.append(ti2)

    def plot_n2(self,ax=None, color='steelblue', start=0, end=np.inf, plot_points=True, alpha=0.05,
          markersize=0.1, linewidth=0.8, ylims=None, kernel_size=3, title=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ind = np.argwhere(np.logical_and(np.array(self.n) > start, np.array(self.n) < end)).ravel()
        N = [self.n[i] for i in ind]

        N2 = [self.n2[i] for i in ind]

        if plot_points:
            for i, j in enumerate(N):
                ax.plot([j] * len(N2[i]), N2[i], '.',
                         color=color, markersize=markersize)

        bound_max = medfilt([np.quantile(ratio_n, 1 - alpha/2.) for ratio_n in N2], kernel_size=kernel_size)
        bound_min = medfilt([np.quantile(ratio_n, alpha/2.) for ratio_n in N2], kernel_size=kernel_size)
        ax.plot(N, bound_min, c=color, linewidth=linewidth, linestyle='-')
        ax.plot(N, bound_max, c=color, linewidth=linewidth, linestyle='-')
        ax.set_xlim([min(N), max(N)])
        ax.set_xlabel(r'$N_1$ ', fontsize=14)
        ax.set_ylabel(r'$N_2$', fontsize=14)
        ax.set_title(title)
        ax.set_ylim(ylims)

        return ax

    def plot_ratio(self, color='steelblue', start=0, end=np.inf, plot_points=True, alpha=0.05,
          markersize=0.1, linewidth=0.8, ylims=None, kernel_size=3, low_cutoff=-2, title=None):

        ind = np.argwhere(np.logical_and(np.array(self.n) > start, np.array(self.n) < end)).ravel()
        N = [self.n[i] for i in ind]

        Ratio = [self.ratio[i] for i in ind]
        Ratio = [[10**i if i != low_cutoff else 0 for i in ratio] for ratio in Ratio]

        if plot_points:
            for i, j in enumerate(N):
                plt.plot([j] * len(Ratio[i]), Ratio[i], '.',
                         color=color, markersize=markersize)

        bound_max = medfilt([np.quantile(ratio_n, 1 - alpha/2.) for ratio_n in Ratio], kernel_size=kernel_size)
        bound_min = medfilt([np.quantile(ratio_n, alpha/2.) for ratio_n in Ratio], kernel_size=kernel_size)
        plt.plot(N, bound_min, c=color, linewidth=linewidth, linestyle='-')
        plt.plot(N, bound_max, c=color, linewidth=linewidth, linestyle='-')

        plt.xlabel(r'Number of Earthquakes $N_1$ ')
        plt.ylabel(r'$\dfrac{N_2}{N_1}$')
        plt.title(title)
        plt.ylim(ylims)
        plt.tight_layout()

    def plot_logratio(self, ax=None, color='steelblue', start=0, end=np.inf, plot_points=True, alpha=0.05,
                      markersize=0.1, linewidth=0.8, ylims=(-2., 2.), kernel_size=3, title=None):

        ind = np.argwhere(np.logical_and(np.array(self.n) > start, np.array(self.n) < end)).ravel()
        N = [self.n[i] for i in ind]

        Ratio = [self.ratio[i] for i in ind]

        if ax is None:
            fig, ax = plt.subplots(1, 1)


        if plot_points:
            for i, j in enumerate(N):
                ax.plot([j] * len(Ratio[i]), Ratio[i], '.',
                         color=color, markersize=markersize)


        bound_max = medfilt([np.quantile(ratio_n, 1 - alpha/2.) for ratio_n in Ratio], kernel_size=kernel_size)
        bound_min = medfilt([np.quantile(ratio_n, alpha/2.) for ratio_n in Ratio], kernel_size=kernel_size)


        ax.plot(N, bound_min, c=color, linewidth=linewidth, linestyle='-')
        ax.plot(N, bound_max, c=color, linewidth=linewidth, linestyle='-')

        ax.set_xlabel(r'$N_1$ ', fontsize=14)
        ax.set_ylabel(r'$\mathrm{log}_{10}\, \dfrac{N_2}{N_1}$', fontsize=14)
        ax.set_title(title)
        ax.set_ylim(ylims)
        ax.set_xlim([min(N), max(N)])

        return ax

    def plot_histogram(self, n, ax=None, bins=50, range_=None, label='Results', ratio=False, color=None):



        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if ratio:
            Y = self.ratio[n]
        else:
            Y = self.n2[n]

        if not range_:
            range_ = (np.min(Y), np.max(Y))
        ax.hist(Y, bins=bins, range=range_, alpha=0.3, label=label, density=True, color=color)
        ax.set_ylabel('$\mathrm{PMF}$')
        ax.set_xlabel(r'$N_2$')

        return range_

    def get_stats(self):

        mean =[]
        median = []
        mode = []
        var = []
        skewness = []
        cod = []

        for i, n in enumerate(self.n):
            mean.append(np.mean(self.n2[i]))
            median.append(np.median(self.n2[i]))
            mode.append(st.mode(self.n2[i])[0])
            var.append(np.var(self.n2[i]))
            cod.append(np.var(self.n2[i])/np.mean(self.n2[i]))
            skewness.append(st.skew(self.n2[i]))

        self.stats = {'mean': mean,
                      'median': median,
                      'mode': mode,
                      'var': var,
                      'skewness': skewness,
                      'cod': cod}

    def plot_stats(self, stat='mean', ax=None, label='', color='steelblue',
                   linestyle='-', linewidth=0.5):

        n = getattr(self, 'n')
        if stat == 'std':
            value = np.sqrt(self.stats.get('var'))
        else:
            value = self.stats.get(stat)

        if ax is None:
            plt.plot(n, value, label=label, color=color, linestyle=linestyle,
                     linewidth=linewidth)
        else:
            ax.plot(n, value, label=label, color=color, linestyle=linestyle,
                    linewidth=linewidth)

    def purge(self):

        self.n2 = []
        self.ratio = []
        self.dt = []
        self.t1 = []
        self.t2 = []
        self.ti1 = []
        self.ti2 = []


class CatalogAnalysis(object):

    def __init__(self, catalog, name=None, params={}):


        self.params = {'n_disc': np.arange(1, 200, 1),
                       'max_iters': 500,
                       'random_time': False,
                       'nonoverlap': False,
                       'ratio_cutoff': -2}
        self.params.update(params)
        self.name = name if name else catalog.name
        self.catalog = catalog
        self.cat_var = CatalogVariability()

    def __eq__(self, other):
        return self.name == other.name and self.params == other.params

    @staticmethod
    def buffer(t1, t2, pdf='uniform'):
        if pdf == 'uniform':
            if isinstance(t1, float):
                return t1 + (t2 - t1) * np.random.random()
            elif isinstance(t1, np.ndarray):
                assert t1.shape == t2.shape
                return t1 + (t2 - t1) * np.random.random(size=t1.shape)

    def get_ratevar(self, verbose=True):

        times = self.catalog.get_decimal_time() #* 365*24*60*60
        n_events = self.catalog.get_number_events()
        max_iters = self.params['max_iters']
        ratio_cutoff = self.params['ratio_cutoff']


        for n in self.params['n_disc']:

            if (n - 1) % 50 == 0 and verbose:
                print('Number of earthquakes %i' % n)
            indices = random.sample(range(1, n_events-n), k=min(n_events-n-1, max_iters))
            indices = np.sort(indices)

            t1 = np.stack((self.buffer(times[indices - 1], times[indices]),
                           self.buffer(times[indices + n - 1], times[indices + n]))).T
            dt = t1[:, 1] - t1[:, 0]

            ti1 = [times[(times > t1_k[0]) & (times <= t1_k[1])] for t1_k in t1]
            t2 = np.stack((t1[:, 1], t1[:, 1] + dt)).T
            ti2 = [times[(times > t2_k[0]) & (times <= t2_k[1])] for t2_k in t2]


            # n2 = [len(ti2_i) for ti2_i in ti2]

            n2 = [len(ti2_i)*dt_i**0.0 for ti2_i, dt_i in zip(ti2, dt)]
            # print(dt_1.shape, np.sum(dt_1))
            ratio = np.array([np.log10(n2_i/n) if n2_i != 0 else ratio_cutoff for n2_i in n2])
            # n2 = [i for i in n2 if i!=0]
            self.cat_var.add(n, n2, ratio, dt, t1, t2, ti1, ti2)
            # self.cat_var.add(n, [n2[i] for i in dt_1 if not i], [ratio[i] for i in dt_1 if i] , dt, t1, t2, ti1, ti2)

    def get_ratevar_time(self):

        times = self.catalog.get_decimal_time()
        ratio_cutoff = self.params['ratio_cutoff']
        time_range = self.catalog.end_year - self.catalog.start_year + 1
        intervals_disc = np.arange(0.0001, 4, 0.0005)

        for t in intervals_disc:
            n_sub = 1
            for j in range(n_sub):
                t_intervals = np.arange(times.min() + t/(j+1), times.max(), t)
                disc = np.digitize(times, t_intervals)
                n, count = np.unique(disc, return_counts=True)
                n1 = count[1:]
                n2 = count[:-1]

                ratio = []
                for n1_i, n2_i in zip(n1, n2):
                    if n1_i == 0 and n2_i !=0:
                        ratio.append(-ratio_cutoff)
                    elif n2_i == 0 and n1_i != 0:
                        ratio.append(ratio_cutoff)
                    elif n2_i == 0 and n1_i == 0:
                        pass
                    else:
                        ratio.append(np.log10(n2_i / n1_i))

                for n_i, n2_i, ratio_i in zip(n1, n2,ratio):
                   self.cat_var.add(n_i, n2_i, ratio_i, dt=np.diff(t_intervals)[0])

        n, ind = np.unique(self.cat_var.n, return_inverse=True)
        self.cat_var.n = n
        n2 = []
        ratio = []

        for ind_i in range(n.shape[0]):
            n2.append(np.array(self.cat_var.n2)[np.argwhere(ind == ind_i)].ravel())
            ratio.append(np.array(self.cat_var.ratio)[np.argwhere(ind == ind_i)].ravel())
        self.cat_var.n2 = n2
        self.cat_var.ratio = ratio

    def get_ratevar_per_tau(self, t):

        times = self.catalog.get_decimal_time()
        ratio_cutoff = self.params['ratio_cutoff']
        time_range = self.catalog.end_year - self.catalog.start_year + 1


        n_sub = 1
        for j in range(n_sub):
            t_intervals = np.arange(times.min() + t / (j + 1), times.max(), t)
            disc = np.digitize(times, t_intervals)
            n, count = np.unique(disc, return_counts=True)
            n1 = count[1:]
            n2 = count[:-1]

            ratio = []
            for n1_i, n2_i in zip(n1, n2):
                if n1_i == 0 and n2_i != 0:
                    ratio.append(-ratio_cutoff)
                elif n2_i == 0 and n1_i != 0:
                    ratio.append(ratio_cutoff)
                elif n2_i == 0 and n1_i == 0:
                    pass
                else:
                    ratio.append(np.log10(n2_i / n1_i))

            for n_i, n2_i, ratio_i in zip(n1, n2, ratio):
                self.cat_var.add(n_i, n2_i, ratio_i, dt=np.diff(t_intervals)[0])

        n, ind = np.unique(self.cat_var.n, return_inverse=True)
        self.cat_var.n = n
        n2 = []
        ratio = []

        for ind_i in range(n.shape[0]):
            n2.append(np.array(self.cat_var.n2)[np.argwhere(ind == ind_i)].ravel())
            ratio.append(np.array(self.cat_var.ratio)[np.argwhere(ind == ind_i)].ravel())
        self.cat_var.n2 = n2
        self.cat_var.ratio = ratio



    @classmethod
    def load(cls, filename=None):
        """
        Loads a serialized Model_results object
        :param filename:
        :return:
        """

        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def save(self, filename=None):
        """
        Serializes Model_results object into a file
        :param filename: If None, save in results folder named with self.name
        """
        if filename:
            with open(filename, 'wb') as obj:
                pickle.dump(self, obj)
        else:
            filename = paths.get('temporal', 'serial', self.name)
            with open(filename, 'wb') as obj:
                pickle.dump(self, obj)
        return filename


class Model(object):   #todo models are replacing themselves due to class inheritance

    def __init__(self, name, params_global={}, params_local={}):
        self.name = name
        self.params_global = params_global
        self.params_local = params_local
        self.sim_var = CatalogVariability()

    @classmethod
    def load(cls, model_name=None, filename=None):
        """
        Loads a serialized Model_results object
        :param filename:
        :return:
        """

        if filename:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        else:
            with open(paths.get('temporal', 'serial', model_name), 'rb') as f:
                obj = pickle.load(f)
            return obj

    def save(self, filename=None):
        """
        Serializes Model object into a file
        :param filename: If None, save in results folder named with self.name
        """
        if filename:
            with open(filename, 'wb') as obj:
                pickle.dump(self, obj)
        else:
            filename = paths.get('temporal', 'serial', self.name)
            with open(filename, 'wb') as obj:
                pickle.dump(self, obj)
        return filename


class forwardPoisson(Model):

    def __init__(self, name=None, metaparams=None):

        self.metaparams = {'n_disc': np.arange(1, 200, 1),
                           'sim_iters': 500,
                           'ratio_cutoff': -2}
        if metaparams:
            self.metaparams.update(metaparams)
        super().__init__(name if name else 'fPoisson')

    def get_params(self, catalog_analysis):

        self.metaparams['n_disc'] = catalog_analysis.params['n_disc']
        self.name += '_' + catalog_analysis.name if self.name == 'fPoisson' else ''

        for i, N in enumerate(catalog_analysis.cat_var.n):
            if (N - 1) % 50 == 0:
                print('N_1: %i' % (N - 1))
            self.params_local[N] = np.mean(catalog_analysis.cat_var.n2[i])

    def simulate(self, nsims=None):

        if not nsims:
            nsims = self.metaparams['sim_iters']

        for n1, params in self.params_local.items():
            N2 = np.random.poisson(params, size=nsims)
            ratio = np.array([np.log10(n2_i / n1) if n2_i != 0 else self.metaparams['ratio_cutoff']
                              for n2_i in N2])

            #### Pending Interarrival times and windows
            self.sim_var.add(n1, n2=N2, ratio=ratio)

    def get_conditional_numbers(self, n, nsims=1000):

        n_events = st.poisson.rvs(self.params_local[n], size=nsims)
        return n_events

    def extend_params(self, n_max, n_cut, n_min=1, deg=1):


        n = self.metaparams['n_disc']
        mu = np.array([self.params_local[i] for i in n])

        id_learn = np.logical_and(n >= n_min, n <= n_cut)

        new_n = np.arange(min(n), n_max)

        poly_mu = np.polyfit(np.sqrt(n[id_learn]), mu[id_learn], deg=deg)
        new_mu = np.polyval(poly_mu, np.sqrt(new_n))


        self.metaparams['n_disc'] = new_n

        for i, n in enumerate(new_n):
            if not (n in self.params_local.keys()):
                self.params_local[n] = [None, None, None, None]

            if n > n_min and new_mu[i] >= 0:
                self.params_local[n] = new_mu[i]


class backwardPoisson(Model):


    def __init__(self, name=None, metaparams={}):

        self.metaparams = {'n_disc': np.arange(1, 200, 1),
                           'sim_iters': 500,
                           'ratio_cutoff': -2}
        self.metaparams.update(metaparams)
        if name:
            super().__init__(name)
        else:
            name = 'bPoisson'
            super().__init__(name)

    def get_params(self, catalog_analysis):

        self.metaparams['n_disc'] = catalog_analysis.params['n_disc']
        self.name += '_' + catalog_analysis.name if self.name == 'bPoisson' else ''

        for i, N in enumerate(catalog_analysis.cat_var.n):
            if (N - 1) % 50 == 0:
                print('N_1: %i' % (N - 1))
            self.params_local[N] = N

    def simulate(self, nsims=None):

        for n1, params in self.params_local.items():

            N2 = np.random.poisson(params, size=self.metaparams['sim_iters'])
            ratio = np.array([np.log10(n2_i / n1) if n2_i != 0 else self.metaparams['ratio_cutoff']
                              for n2_i in N2])

            self.sim_var.add(n1, n2=N2, ratio=ratio)

    def get_conditional_numbers(self, n, nsims=1000):

        n_events = st.poisson.rvs(n, size=nsims)
        return n_events


class NegBinom(Model):
    """
    Class with Local Negative Binomial (fit for every N1)
    Parameters are [tau, theta, lambda, alpha]
    where tau/theta are both wikipedia and scipy parameters, and
    lambda/alpha are parameters used in Kagan, 2010  Eq.18
    """


    def __init__(self, name=None, metaparams={}):

        self.metaparams = {'n_disc': np.arange(1, 200, 1),
                           'sim_iters': 500,
                           'ratio_cutoff': -2,
                           'n_params': 4}
        self.metaparams.update(metaparams)

        if name:
            super().__init__(name)
        else:
            name = 'negbinom'
            super().__init__(name)

    def get_params(self, catalog_analysis):

        self.metaparams['n_disc'] = catalog_analysis.params['n_disc']
        self.name += '_' + catalog_analysis.name if self.name == 'negbinom' else ''

        for i, N in enumerate(catalog_analysis.cat_var.n):

            if (N - 1) % 50 == 0:
                print('N_1: %i' % (N - 1))

            n2 = catalog_analysis.cat_var.n2[i]

            N2 = [i for i in n2 if i != 0]

            Fit = statsmodels.discrete.discrete_model.NegativeBinomial(
                        N2, np.ones(len(N2)), loglike_method='nb2')
            res = Fit.fit_regularized(disp=False)

            lambda_ = np.exp(res.params[0])
            alpha = res.params[1]
            #todo todo todoooo
            # Q = 1 if self.metaparams['loglike_method'] == 'nb1' else 0
            Q = 0
            tau = 1. / alpha * lambda_ ** Q
            theta = tau / (tau + lambda_)
            self.params_local[N] = [tau, theta, lambda_, alpha / lambda_ ** Q]

    def plot_params(self, ax2=None):

        if ax2 is None:
            fig, ax2 = plt.subplots()


        n = self.metaparams['n_disc']
        lambda_ = np.array([self.params_local[i][2] for i in n])
        alpha = np.array([self.params_local[i][3] for i in n])

        ax2.plot(n, alpha, '.', color='blueviolet', label=r'$\alpha$ data')
        ax2.set_ylabel(r'Dispersion $\alpha$', color='blueviolet', fontsize=14)
        ax2.set_xlabel(r'$N_1$', fontsize=14)
        ax = ax2.twinx()
        ax.plot(n, lambda_, '.', color='brown', label='$\mu$ data')
        ax.set_ylabel(r'$\mu=\mathrm{E}[N]$', color='brown', fontsize=14)

        return ax, ax2

    def extend_params(self, n_max, n_cut, n_min=1, limit=True, parameters='nb2', deg=1, axes=None, plot=True):


        n = self.metaparams['n_disc']
        if parameters == 'nb2':
            mu = np.array([self.params_local[i][2] for i in n])
            alpha = np.array([self.params_local[i][3] for i in n])

            id_learn = np.logical_and(n >= n_min, n <= n_cut)

            new_n = np.arange(min(n), n_max)

            poly_mu = np.polyfit(np.sqrt(n[id_learn]), mu[id_learn], deg=2)
            new_mu = np.polyval(poly_mu, np.sqrt(new_n))



            poly_alpha = np.polyfit(np.log(n[id_learn]), alpha[id_learn], deg=2)
            new_alpha = np.polyval(poly_alpha, np.log(new_n))

            new_alpha[new_alpha<0.01] = 0.01
            if limit:
                for i, j in enumerate(new_n):
                    if new_mu[i] < j:
                        print(new_mu, j)
                        new_mu[i] = j



            if plot:
                if axes is None:
                    fig, ax2 = plt.subplots()

                    ax2.plot(new_n, new_alpha, '.', color='blueviolet', label=r'$\alpha$ fit')
                    ax2.set_ylabel(r'$\alpha$', color='blueviolet', fontsize=14)
                    ax2.set_xlabel(r'$N_1$', fontsize=14)
                    ax = ax2.twinx()
                    ax.plot(new_n, new_mu, '.', color='brown', label='$\mu$ data')
                    ax.set_ylabel(r'Mean rate $\mu$', color='brown', fontsize=14)
                    axes = [ax, ax2]
                else:
                    axes[0].plot(new_n, new_mu, color='brown', linestyle='--', label='$\mu$ fit')
                    axes[1].plot(new_n, new_alpha, color='purple', linestyle='--', label=r'$\alpha$ fit')

                axes[0].legend(loc=1)
                axes[1].legend(loc=4)

            self.metaparams['n_disc'] = new_n
            Q = 1 if parameters == 'nb1' else 0
            for i, n in enumerate(new_n):
                if not (n in self.params_local.keys()):
                    self.params_local[n] = [None, None, None, None]

                if n > n_min and new_mu[i] >= 0:
                    self.params_local[n][2] = new_mu[i]
                self.params_local[n][3] = new_alpha[i]

                tau = 1. / self.params_local[n][3] * self.params_local[n][2] ** Q
                theta = tau / (tau + self.params_local[n][2])
                self.params_local[n][0] = tau
                self.params_local[n][1] = theta



    def simulate(self, nsims=None):
        if not nsims:
            nsims = self.metaparams['sim_iters']
        for n1, params in self.params_local.items():
            n2 = [i for i in st.nbinom.rvs(params[0], params[1], size=nsims) if i != 0]
            ratio = np.array([np.log10(n2_i / n1) if n2_i != 0 else self.metaparams['ratio_cutoff']
                              for n2_i in n2])
            self.sim_var.add(n1, n2=n2, ratio=ratio)


    def get_conditional_numbers(self, n, nsims=1000):

        n_events = st.nbinom.rvs(self.params_local[n][0], self.params_local[n][1], size=nsims)
        return n_events


class MixedPoisson(Model):

    """

    """

    def __init__(self, name=None, metaparams={}):

        self.metaparams = {'n_disc': np.arange(1, 200, 1),
                           'sim_iters': 1000,
                           'sim_subiters': 400,
                           'ratio_cutoff': -2,
                           'distribution': 'lognorm'}
        self.metaparams.update(metaparams)

        if name:
            super().__init__(name)
        else:
            name = '%s-Poisson' % self.metaparams['distribution']
            super().__init__(name)

    def get_params(self, catalog_analysis):

        self.metaparams['n_disc'] = catalog_analysis.params['n_disc']
        self.name += '_' + catalog_analysis.name if self.name == '%s-Poisson' % self.metaparams['distribution'] else ''

        for i, N in enumerate(catalog_analysis.cat_var.n):
            if (N - 1) % 50 == 0:
                print('N_1: %i' % (N - 1))

            n2 = [i if i != 0 else 1 for i in catalog_analysis.cat_var.n2[i]]
            dist = getattr(st, self.metaparams['distribution'])

            if self.metaparams['distribution'] in ('lognorm', 'gamma'):
                params = dist.fit(n2, floc=0)

            elif self.metaparams['distribution'] == 'beta':
                params = dist.fit(n2, floc=0, fscale=(np.max(n2) + 1)) #𝛼, 𝛽, loc (lower limit), scale (upper limit - lower limit)

            self.params_local[N] = list(params)

    def plot_params(self, ax2=None, beta_params=(0, 1)):

        if self.metaparams['distribution'] == 'lognorm':
            if ax2 is None:
                fig, ax2 = plt.subplots()

            n = self.metaparams['n_disc']
            mu = np.array([self.params_local[i][2] for i in n])
            var = np.array([self.params_local[i][0] for i in n])

            ax2.plot(n, var, '.', color='blueviolet', label=r'$\sigma_x$ data')
            ax2.set_ylabel(r'$\sigma_x$', color='blueviolet', fontsize=14)
            ax2.set_xlabel(r'$N_1$', fontsize=14)
            ax = ax2.twinx()
            ax.plot(n, mu, '.', color='brown', label=r'$\mu_x$ data')
            ax.set_ylabel(r'$\mu_x=\mathrm{Med}[N]$', color='brown', fontsize=14)

        elif self.metaparams['distribution'] == 'gamma':
            if ax2 is None:
                fig, ax2 = plt.subplots()

            n = self.metaparams['n_disc']
            tau = np.array([self.params_local[i][0] for i in n])
            p = np.array([self.params_local[i][2] for i in n])          # theta = 1 / (p + 1)


            ax2.plot(n, tau, '.', color='blueviolet', label=r'$\tau data')
            ax2.set_ylabel(r'$\tau$', color='blueviolet', fontsize=14)
            ax2.set_xlabel(r'$N_1$', fontsize=14)
            ax = ax2.twinx()
            ax.plot(n, p, '.', color='brown', label=r'$p=\frac{1-\theta}{\theta$ data')
            ax.set_ylabel(r'$p$', color='brown', fontsize=14)

        elif self.metaparams['distribution'] == 'beta':
            if ax2 is None:
                fig, ax2 = plt.subplots()

            param_names = [r'$\alpha$', r'$\beta$', r'location', r'scale']
            n = self.metaparams['n_disc']
            param_1 = np.array([self.params_local[i][beta_params[0]] for i in n])
            param_2 = np.array([self.params_local[i][beta_params[1]] for i in n])          # theta = 1 / (p + 1)

            ax2.plot(n, param_2, '.', color='blueviolet', label=f'{param_names[beta_params[1]]} data')
            ax2.set_ylabel(f'{param_names[beta_params[1]]}', color='blueviolet', fontsize=14)
            ax2.set_xlabel(r'$N_1$', fontsize=14)
            ax = ax2.twinx()
            ax.plot(n, param_1, '.', color='brown', label=f'{param_names[beta_params[0]]} data')
            ax.set_ylabel(f'{param_names[beta_params[0]]}', color='brown', fontsize=14)


        return ax, ax2

    def extend_params(self, n_max, n_cut, n_min=1, axes=None, plot=True, beta_params=(0, 1)):

        if self.metaparams['distribution'] == 'lognorm':
            print('AAA')
            n = self.metaparams['n_disc']
            mu = np.array([self.params_local[i][2] for i in n])
            std = np.array([self.params_local[i][0] for i in n])

            id_learn = np.logical_and(n >= n_min, n <= n_cut)

            new_n = np.arange(min(n), n_max)

            poly_mu = np.polyfit(n[id_learn], mu[id_learn], deg=1)
            new_mu = np.polyval(poly_mu, new_n)

            poly_std = np.polyfit(np.log(n[id_learn]), std[id_learn], deg=2)
            new_std = np.polyval(poly_std, np.log(new_n))
            self.metaparams['n_disc'] = new_n
            for i, m in enumerate(new_n):
                if not (m in self.params_local.keys()):
                    self.params_local[m] = [None, 0, None]
                self.params_local[m][0] = new_std[i]
                self.params_local[m][2] = new_mu[i]

            if plot:
                if axes is None:
                    fig, ax2 = plt.subplots()
                    ax2.plot(new_n, new_std, '-', color='blueviolet', label=r'$\sigma_x$ fit')
                    ax2.set_ylabel(r'$\sigma_x', color='blueviolet', fontsize=14)
                    ax2.set_xlabel(r'N_1', fontsize=14)
                    ax = ax2.twinx()
                    ax.plot(new_n, new_mu, '-', color='brown', label=r'$\mu_x$ data')
                    ax.set_ylabel(r'$\mu_x$', color='brown', fontsize=14)
                    axes = (ax, ax2)

                else:
                    axes[0].plot(new_n, new_mu, color='brown', linestyle='--', label=r'$\xi$ fit')
                    axes[1].plot(new_n, new_std, color='purple', linestyle='--', label=r'$\sigma$ fit')

                axes[0].legend(loc=1).set_zorder(102)
                axes[1].legend(loc=4).set_zorder(102)


        elif self.metaparams['distribution'] == 'beta':
            n = self.metaparams['n_disc']
            alpha = np.array([self.params_local[i][0] for i in n])
            beta = np.array([self.params_local[i][1] for i in n])
            scale = np.array([self.params_local[i][3] for i in n])


            id_learn = np.logical_and(n >= n_min, n <= n_cut)

            new_n = np.arange(min(n), n_max)

            poly_alpha = np.polyfit(np.log(n[id_learn]), alpha[id_learn], deg=2)
            new_alpha = np.polyval(poly_alpha, np.log(new_n))

            poly_beta = np.polyfit(np.log(n[id_learn]), beta[id_learn], deg=3)
            new_beta = np.polyval(poly_beta, np.log(new_n))

            poly_scale = np.polyfit(np.log(n[id_learn]), scale[id_learn], deg=3)
            new_scale = np.polyval(poly_scale, np.log(new_n))

            self.metaparams['n_disc'] = new_n
            for i, m in enumerate(new_n):
                if not (m in self.params_local.keys()):
                    self.params_local[m] = [None, None, 0, None]
                self.params_local[m][0] = new_alpha[i]
                self.params_local[m][1] = new_beta[i]
                self.params_local[m][3] = new_scale[i]

            if plot:
                param_names = [r'$\alpha$', r'$\beta$', r'location', r'scale']
                param_1 = np.array([self.params_local[i][beta_params[0]] for i in new_n])
                param_2 = np.array([self.params_local[i][beta_params[1]] for i in new_n])  # theta = 1 / (p + 1)

                if axes is None:

                    fig, ax2 = plt.subplots()
                    ax2.plot(new_n, param_2, '.', color='blueviolet', label=f'{param_names[beta_params[1]]} fit')
                    ax2.set_ylabel(f'{param_names[beta_params[1]]}', color='blueviolet', fontsize=14)
                    ax2.set_xlabel(r'n', fontsize=14)
                    ax = ax2.twinx()
                    ax.plot(new_n, param_1, '.', color='brown', label=f'{param_names[beta_params[0]]} fit')
                    ax.set_ylabel(f'{param_names[beta_params[0]]}', color='brown', fontsize=14)
                    axes = (ax, ax2)

                else:
                    axes[0].plot(new_n, param_1, color='brown', linestyle='--',  label=f'{param_names[beta_params[1]]} fit')
                    axes[1].plot(new_n, param_2, color='purple', linestyle='--', label=f'{param_names[beta_params[0]]} fit')

                axes[0].legend(loc=1)
                axes[1].legend(loc=4)


    def simulate(self, nsims=1000, mu_sims=None, number_sims=None):

        if mu_sims is None:
            mu_sims = self.metaparams['sim_iters']
        if number_sims is None:
            number_sims = self.metaparams['sim_subiters']

        for n1, params in self.params_local.items():

            dist = getattr(st, self.metaparams['distribution'])
            mu = dist.rvs(*params, size=mu_sims)
            n2_full = np.array([st.poisson.rvs(i, size=number_sims) for i in mu]).ravel()
            n2 = np.random.choice(n2_full, size=nsims)
            ratio = np.array([np.log10(n2_i / n1) if n2_i != 0 else self.metaparams['ratio_cutoff']
                              for n2_i in n2])
            #### Pending Interarrival times and windows
            self.sim_var.add(n1, n2=n2, ratio=ratio)


    def get_conditional_numbers(self, n, nsims=1000):
        dist = getattr(st, self.metaparams['distribution'])
        mu_ln = dist.rvs(*self.params_local[n], size=self.metaparams['sim_iters'])
        n_full = np.ravel([st.poisson.rvs(i, size=int(0.4 * self.metaparams['sim_subiters'])) for i in mu_ln])
        n_events = random.choices(n_full, k=nsims)

        return n_events


class GlobalPoisson(Model):


    def __init__(self, name=None, metaparams={}):


        self.metaparams = {'n_disc': np.arange(1, 200, 1),
                           'sim_iters': 500,
                           'ratio_cutoff': -2,
                           'n_params': 4}
        self.metaparams.update(metaparams)
        if name:
            super().__init__(name)
        else:
            name = 'gPoisson'
            super().__init__(name)

    def get_params(self, catalog_analysis, time_interval=1):

        self.metaparams['n_disc'] = catalog_analysis.params['n_disc']
        self.name += '_' + catalog_analysis.name if self.name == 'gPoisson' else ''

        times = catalog_analysis.catalog.get_decimal_time()
        start_time = catalog_analysis.catalog.start_year
        end_time = catalog_analysis.catalog.end_year
        time_discretization = np.arange(start_time, end_time, time_interval)
        time_counts = np.digitize(times, time_discretization)
        time, count = np.unique(time_counts, return_counts=True)
        rate_global = np.mean(count)

        self.params_global['rate'] = rate_global

        for i, N in enumerate(catalog_analysis.cat_var.n):

            dt = catalog_analysis.cat_var.dt[i]
            rate_i = np.mean(dt*rate_global/time_interval)
            self.params_local[N] = rate_i

    def simulate(self, nsims=None):

        for n1, params in self.params_local.items():

            N2 = np.random.poisson(params, size=self.metaparams['sim_iters'])
            ratio = np.array([np.log10(n2_i / n1) if n2_i != 0 else self.metaparams['ratio_cutoff']
                              for n2_i in N2])

            #### Pending Interarrival times and windows
            self.sim_var.add(n1, n2=N2, ratio=ratio)

    def get_conditional_numbers(self, n, nsims=1000):

        n_events = st.poisson.rvs(self.params_local[n], size=nsims)
        return n_events


class GlobalNegBinom(Model):


    def __init__(self, name=None, metaparams={}):

        self.metaparams = {'n_disc': np.arange(1, 200, 1),
                           'sim_iters': 1000,
                           'ratio_cutoff': -2,
                           'n_params': 4}
        self.metaparams.update(metaparams)
        if name:
            super().__init__(name)
        else:
            name = 'gnegbinom'
            super().__init__(name)

    def get_params(self, catalog_analysis, time_interval=1):

        self.metaparams['n_disc'] = catalog_analysis.params['n_disc']
        self.name += '_' + catalog_analysis.name if self.name == 'gnegbinom' else ''

        times = catalog_analysis.catalog.get_decimal_time()
        start_time = catalog_analysis.catalog.start_year
        end_time = catalog_analysis.catalog.end_year
        time_discretization = np.arange(start_time, end_time , time_interval)
        time_counts = np.digitize(times, time_discretization)
        time, count = np.unique(time_counts, return_counts=True)

        m1 = np.mean(count)
        m2 = st.moment(count, 2)

        mu = m1
        alpha = (m2 - m1) / m1 ** 2
        tau = 1. / alpha
        theta = tau/(mu + tau)

        F_nb = [0]
        count_disc = np.arange(0, np.max(count) * 1.3)
        for k in count_disc[1:]:
            F_k = betainc(tau, k, theta)
            F_nb.append(F_k)


        self.params_global['tau'] = tau
        self.params_global['theta'] = theta
        self.params_global['lambda_'] = mu
        self.params_global['alpha'] = alpha

        for i, N in enumerate(catalog_analysis.cat_var.n):

            dt = catalog_analysis.cat_var.dt[i]
            mu_i = N
            factor = N/mu

            #### theta constant
            theta_i = theta
            tau_i = tau*factor
            alpha_i = 1/tau_i

            ### alpha constant
            # alpha_i = alpha
            # # mu_i = mu/factor
            # tau_i = 1/alpha_i
            # theta_i = tau_i/(tau_i + mu_i)

            self.params_local[N] = [tau_i, theta_i, mu_i, alpha_i]

    def simulate(self, nsims=None):
        if not nsims:
            nsims = self.metaparams['sim_iters']

        for n1, params in self.params_local.items():
            N2 = np.random.negative_binomial(*params[:2], size=nsims)
            ratio = np.array([np.log10(n2_i / n1) if n2_i != 0 else self.metaparams['ratio_cutoff']
                              for n2_i in N2])
            self.sim_var.add(n1, n2=N2, ratio=ratio)

    def get_conditional_numbers(self, n, nsims=1000):

        n_events = st.nbinom.rvs(self.params_local[n][0], self.params_local[n][1], size=nsims)
        return n_events


class Empirical(Model):



    def __init__(self, name=None, metaparams={}):

        self.metaparams = {'n_disc': np.arange(1, 200, 1),
                           'sim_iters': 500,
                           'ratio_cutoff': -2,
                           'loglike_method': 'nb1',
                           'n_params': 4}
        self.metaparams.update(metaparams)

        if name:
            super().__init__(name)
        else:
            name = 'empirical'
            super().__init__(name)


    def get_params(self, catalog_analysis, verbose=True):

        self.metaparams['n_disc'] = catalog_analysis.params['n_disc']
        self.name += '_' + catalog_analysis.name if self.name == 'empirical' else ''

        for i, N in enumerate(catalog_analysis.cat_var.n):
            if verbose:
                if (N - 1) % 50 == 0:
                    print('N_1: %i' % (N - 1))

            n2 = catalog_analysis.cat_var.n2[i]

            N2 = n2

            A = ECDF(N2)
            x = np.arange(max(A.x))
            cdf_x = np.interp(x, A.x, A.y)
            pdf = np.diff(cdf_x)
            pdf /= np.sum(pdf)
            self.params_local[N] = pdf


    def simulate(self, nsims=None):
        if nsims is None:
            nsims = self.metaparams['sim_iters']
        for n1, params in self.params_local.items():
            distribution = st.rv_discrete(values=(np.arange(len(params)), params))
            n2 = distribution.rvs(size=nsims)
            ratio = np.array([np.log10(n2_i / n1) if n2_i != 0 else self.metaparams['ratio_cutoff']
                              for n2_i in n2])
            self.sim_var.add(n1, n2=n2, ratio=ratio)


    def get_conditional_numbers(self, n, nsims=1000):
        params = self.params_local[n]
        distribution = st.rv_discrete(values=(np.arange(len(params)), params))
        n_events = distribution.rvs(size=nsims)

        return n_events


def create_models(cat):

    analysis = CatalogAnalysis(cat, name=cat.name, params={'n_disc': np.arange(1, 500), 'max_iters': 1500})
    analysis.get_ratevar()
    analysis.save()

    negbinom = NegBinom()
    negbinom.get_params(analysis)
    negbinom.extend_params(1250, 500, 20)
    negbinom.save()

    lognorm = MixedPoisson(metaparams={'distribution': 'lognorm'})
    lognorm.get_params(analysis)
    lognorm.extend_params(1250, 500, 20)
    lognorm.save()

    gamma = MixedPoisson(metaparams={'distribution': 'gamma'})
    gamma.get_params(analysis)
    gamma.save()

    beta = MixedPoisson(metaparams={'distribution': 'beta'})
    beta.get_params(analysis)
    beta.save()

    gPoisson = GlobalPoisson()
    gPoisson.get_params(analysis)
    gPoisson.save()

    bPoisson = backwardPoisson()
    bPoisson.get_params(analysis)
    bPoisson.save()

    fPoisson = forwardPoisson()
    fPoisson.get_params(analysis)
    fPoisson.save()

    gnegbinom = GlobalNegBinom(metaparams={'sim_iters': 1000, 'loglike_method': 'nb1'})
    gnegbinom.get_params(analysis)
    gnegbinom.save()

    empirical = Empirical()
    empirical.get_params(analysis)
    empirical.save()


if __name__ == '__main__':

    cat = catalogs.get_cat_nz()
    cat = catalogs.filter_cat(cat, mws=(4.0, 10.0), start_time=dt(1960, 1, 1), shapefile=paths.region_nz_collection)
    a = create_models(cat)




