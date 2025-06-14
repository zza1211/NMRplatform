"""
Perform cross validation.
"""
import collections
import typing
import numpy as np

from .core import kfold_cv_opls, kfold_cv_pls, kfold_prediction

from . import pretreatment
from .pls import PLS
from .opls import OPLS

import tqdm
from typing import Union

class CrossValidation:
    """
    Stratified cross validation

    Parameters:
    ----------
    estimator: str
        Estimator indicates algorithm for model construction.
        Values can be "pls" for PLS and "opls" for OPLS. Default
        is "opls".
    kfold: int
        k fold cross validation. if k equals to len(X), leave one out
        cross validation will be performed. Default is 10.
    scaler: str
        Scaler for scaling data matrix. Valid values are "uv" for
        zero-mean-unit-variance scaling, "pareto" for Pareto scaling,
        "minmax" for Min-Max scaling and "mean" for mean centering.
        Default is "pareto".
    tol: float
        Tolerance for PLS NIPALS iteration.
    max_iter: int
        Maximum number of iterations for PLS NIPALS iteration.

    Returns
    -------
    CrossValidation object

    """
    def __init__(self, estimator="opls", kfold=10, scaler="pareto",
                 tol=1.e-10, max_iter=1000):
        # number of folds
        self.kfold: int = kfold
        self._tol: float = tol
        self._max_iter: int = max_iter
        self._scaler_param: str = scaler
        self.valid_for_splot: bool = scaler in ("pareto", "mean")
        self.estimator_name: str = estimator
        self.use_opls: bool = estimator == "opls"
        # estimator
        f_estimator, f_scaler = self._create_scaler_estimator()
        self.estimator: Union[PLS, OPLS] = f_estimator
        self.scaler: [pretreatment.Scaler] = f_scaler
        # initialize other attributes, but should be HIDDEN
        self._Tortho: typing.Optional[np.ndarray] = None
        self._Tpred: typing.Optional[np.ndarray] = None
        self._n: typing.Optional[int] = None
        self._p: typing.Optional[int] = None
        self._pcv: typing.Optional[np.ndarray] = None
        self._pcv_ortho: typing.Optional[np.ndarray] = None
        self._cv_num_vars: typing.Optional[np.ndarray] = None
        self._cv_preds: typing.Optional[np.ndarray] = None
        self._opt_component: typing.Optional[int] = None
        self._mis_classifications: typing.Optional[np.ndarray] = None
        self._q2: typing.Optional[np.ndarray] = None
        self._npc0: typing.Optional[int] = None
        self._x: typing.Optional[np.ndarray] = None
        self.y: typing.Optional[np.ndarray] = None
        self.groups: typing.Optional[dict] = None
        self._r2xcorr: typing.Optional[np.ndarray] = None
        self._r2xyo: typing.Optional[np.ndarray] = None
        self._corr_y_perms: typing.Optional[np.ndarray] = None
        self._perm_q2: typing.Optional[np.ndarray] = None
        self._perm_r2: typing.Optional[np.ndarray] = None
        self._perm_err: typing.Optional[np.ndarray] = None
        self._vip: typing.Optional[np.ndarray] = None
        self._coefcs: typing.Optional[np.ndarray] = None
        self._explained_variance_ratio: typing.Optional[np.ndarray] = None

    def fit(self, x, y):
        """
        Fitting variable matrix X

        Parameters
        ----------
        x : np.ndarray
            Variable matrix with size n samples by p variables.
        y : np.ndarray | list
            Dependent matrix with size n samples by 1. The values in
            this vector must be 0 and 1, otherwise the classification
            performance will be wrongly concluded.

        Returns
        -------
        CrossValidation object

        """
        # TODO: Check dimension consistencies between X and y.
        # set the labels in y to 0 and 1, and name the groups using
        # the labels in y
        y = self._reset_y(y)
        x = self._check_x(x)
        scale_tag: int = self.scaler.scaler_tag

        if self.use_opls:
            (pred_y, q2, r2xyo, r2xcorr, no_mcs, t_o, t_p, p_p, p_o,
             n_opt, n0) = kfold_cv_opls(x, y, self.kfold, scale_tag,
                                        self._tol, self._max_iter)
            self._r2xcorr = r2xcorr
            self._r2xyo = r2xyo
            self._Tortho = t_o
            self._pcv_ortho = p_o
        else:
            pred_y, q2, no_mcs, t_p, p_p, n_opt, n0 = kfold_cv_pls(
                x, y, self.kfold, scale_tag, self._tol, self._max_iter)

        self._opt_component = n_opt
        self._mis_classifications = no_mcs
        self._pcv = p_p
        self._cv_preds = pred_y
        # Q2
        self._q2 = q2
        # predictive scores if OPLS is used, or scores if PLS is used
        self._Tpred = t_p
        self._npc0 = n0

        self._n = x.shape[0]
        self._p = x.shape[1]
        self._x = x
        self.y = y

        # refit for a final model
        self._create_optimal_model(x, y)

    def predict(self, x, return_scores=False):
        """
        Does prediction using optimal model.

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by p variables.
        return_scores: bool
            For OPLS, it's possible to return predictive scores. Thus
            setting this True with `estimator` being "opls" will return
            the predictive scores

        Returns
        -------
        y: np.ndarray
            Predictions for the x
        scores: np.ndarray
            Predictive scores for OPLS

        """
        # TODO: check the dimension consistencies between the training
        #       data and the input data matrix.
        npc = self._opt_component + 1
        # scale the matrix
        x = self.scaler.scale(x[:, self.scaler.variable_index])
        if self.use_opls:
            x = self.estimator.correct(x.copy(), n_component=npc)
            return self.estimator.predict(
                x, n_component=npc, return_scores=return_scores
            )
        return self.estimator.predict(x, n_component=npc)

    def permutation_test(self, num_perms=200) -> None:
        """
        Performs permutation test on constructed model.

        Parameters
        ----------
        num_perms: int
            Number of permutations. Defaults to 10000.

        Returns
        -------
        None

        """
        # check the arguments
        if not isinstance(num_perms, int):
            raise ValueError("Expected integer, got {}.".format(num_perms))
        if num_perms < 20:
            raise ValueError("Expected large positive integer >= 20, "
                             "got {}.".format(num_perms))

        atag: int = 1 if self.use_opls else 2
        k: int = self.kfold

        # do permutation test
        x = np.ascontiguousarray(self._x[:, self.scaler.variable_index],
                                 np.float64)
        # center y
        y_center = self.y - self.y.mean()
        ssy_c = (y_center ** 2).sum()
        # optimal component number
        npc: int = self._opt_component + 1
        n: int = self.y.size
        scale_tag: int = self.scaler.scaler_tag

        rnd_generator = np.random.default_rng()

        perm_q2: np.ndarray = np.zeros(num_perms, dtype=np.float64)
        perm_r2: np.ndarray = np.zeros(num_perms, dtype=np.float64)
        perm_err: np.ndarray = np.zeros(num_perms, dtype=np.float64)
        perm_corr: np.ndarray = np.zeros(num_perms, dtype=np.float64)
        for i in tqdm.tqdm(range(num_perms), total=num_perms,
                           desc="Calculating permuted metrics"):
            # randomize labels
            ix = rnd_generator.permutation(n)
            ry = self.y[ix]
            q2, r2, err = kfold_prediction(x, ry, k, npc, scale_tag, atag,
                                           self._tol, self._max_iter)
            perm_err[i] = err
            perm_q2[i] = q2
            perm_r2[i] = r2
            perm_corr[i] = (((y_center * y_center[ix]).sum()) / ssy_c) ** 2

        self._perm_q2 = perm_q2
        self._perm_r2 = perm_r2
        self._perm_err = perm_err
        self._corr_y_perms = perm_corr

    def reset_optimal_num_component(self, k) -> None:
        """
        Resets the optimal number of components for manual setup.

        Parameters
        ----------
        k: int
            Number of components according to the error plot.

        Returns
        -------
        None

        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("The number must be a positive integer.")

        if k > self._npc0:
            raise ValueError("The number must not exceed the maximum "
                             f" number of components {self._npc0}.")

        self._opt_component = k
        # re-fit the model using the updated optimal number of components
        self._create_optimal_model(self._x, self.y)

    @property
    def coefcs(self) -> np.ndarray:
        """
        Returns the coefficients of centered and scaled X.

        Returns
        -------
        np.ndarray
            The coefficients of centered and scaled X.

        """
        return self._coefcs

    @property
    def orthogonal_score(self) -> np.ndarray:
        """
        Returns cross validated orthogonal score.

        Returns
        -------
        np.ndarray
            The first orthogonal scores.

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if not self.use_opls:
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self._Tortho[self._opt_component]

    @property
    def predictive_score(self) -> np.ndarray:
        """
        Returns cross validated predictive score.

        Returns
        -------
        np.ndarray
            The first predictive scores.

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if not self.use_opls:
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self._Tpred[self._opt_component]

    @property
    def scores(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray
            The first predictive score, if the method is OPLS/OPLS-DA,
            otherwise is the scores of X

        """
        if self.use_opls:
            return self.predictive_score
        else:
            return self.estimator.scores_x

    @property
    def q2(self) -> float:
        """
        Returns cross validated Q2.

        Returns
        -------
        q2: float

        """
        return float(self._q2[self._opt_component])

    @property
    def r2(self) -> float:
        """
        Returns fitted R2 after determining the optimal number of
        components.

        Returns
        -------
        R2: float

        """
        return self.estimator.r2

    @property
    def optimal_component_num(self) -> int:
        """
        Number of components determined by cross validation.

        Returns
        -------
        int

        """
        return self._opt_component + 1

    @property
    def r2xcorr(self) -> float:
        """
        Returns
        -------
        float
            Modeled joint X-y covariation of X.

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if not self.use_opls:
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return float(self._r2xcorr[self._opt_component])

    @property
    def r2xyo(self) -> float:
        """
        Returns
        -------
        float
            Modeled structured noise variation of X.

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if not self.use_opls:
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return float(self._r2xyo[self._opt_component])

    @property
    def r2x(self) -> float:
        """

        Returns
        -------
        float
            Modeled variation of X

        """
        return float(self.estimator.r2x[self._opt_component])

    @property
    def r2y(self) -> float:
        """

        Returns
        -------
        float
            Modeled variation of y

        """
        return float(self.estimator.r2y[self._opt_component])

    @property
    def r2x_cum(self) -> float:
        """
        Cumulative fraction of the sum of squares explained up to the
        optimal number of principal components.

        Returns
        -------
        float
            Cumulative fraction of the sum of squares explained

        """
        return float(self.estimator.r2x_cum[self._opt_component])

    @property
    def r2y_cum(self) -> float:
        """
        Cumulative fraction of the sum of squares explained up to the
        optimal number of principal components.

        Returns
        -------
        float
            Cumulative fraction of the sum of squares explained

        """
        return float(self.estimator.r2y_cum[self._opt_component])

    @property
    def correlation(self) -> np.ndarray:
        """ Correlation
        Returns
        -------
        np.ndarray
            Correlation loading profile

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        References
        ----------
        [1] Wiklund S, et al. Visualization of GC/TOF-MS-Based
        Metabolomics Data for Identification of Biochemically
        Interesting Compounds Using OPLS Class Models. Anal Chem.
        2008, 80, 115-122.

        """
        if not self.use_opls:
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self.estimator.corr

    @property
    def covariance(self) -> np.ndarray:
        """
        Covariance

        Returns
        -------
        np.ndarray
            Correlation loading profile

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        References
        ----------
        [1] Wiklund S, et al. Visualization of GC/TOF-MS-Based
        Metabolomics Data for Identification of Biochemically
        Interesting Compounds Using OPLS Class Models. Anal Chem.
        2008, 80, 115-122.

        """
        if not self.use_opls:
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self.estimator.cov

    @property
    def orthogonal_loadings_cv(self) -> np.ndarray:
        """
        Orthogonal loadings from cross validation.

        Returns
        -------
        np.ndarray
            Correlation loading profile

        Raises
        ------
        ValueError
            If OPLS / OPLS-DA is not used.

        """
        if not self.use_opls:
            raise ValueError("This is only applicable for OPLS/OPLS-DA.")
        return self._pcv_ortho

    @property
    def loadings_cv(self) -> np.ndarray:
        """
        Loadings from cross validation. If OPLS / OPLS-DA is used, this
        is the predictive loadings.

        Returns
        -------

        """
        return self._pcv

    @property
    def min_nmc(self) -> int:
        """

        Returns
        -------
        int
            Minimal number of mis-classifications obtained by
            cross validation.

        """
        return int(self._mis_classifications[self._opt_component])

    @property
    def mis_classifications(self) -> np.ndarray:
        """

        Returns
        -------
        list
            Mis-classifications at different principal components.

        """
        return self._mis_classifications

    @property
    def used_variable_index(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray:
            Indices of variables used for model construction

        """
        return self.scaler.variable_index

    @property
    def permutation_q2(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray:
            Q2 array generated by permutation test.

        Raises
        ------
        ValueError

        """
        if self._perm_q2 is None:
            raise ValueError("Permutation test has not been performed.")
        return self._perm_q2

    @property
    def permutation_r2(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray:
            R2 array generated by permutation test.

        Raises
        ------
        ValueError

        """
        if self._perm_r2 is None:
            raise ValueError("Permutation test has not been performed.")
        return self._perm_r2

    @property
    def permutation_error(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray:
            Misclassification error rates generated by permutation test.

        Raises
        ------
        ValueError

        """
        if self._perm_err is None:
            raise ValueError("Permutation test has not been performed.")
        return self._perm_err

    @property
    def correlation_permute_y(self) -> np.ndarray:
        """

        Returns
        -------
        np.ndarray:
            Correlation between permuted y and normal y.

        Raises
        ------
        ValueError

        """
        if self._corr_y_perms is None:
            raise ValueError("Permutation test has not been performed.")
        return self._corr_y_perms

    @property
    def vip(self) -> np.ndarray:
        """
        Returns VIPs of variables

        """
        return self._vip

    @property
    def num_samples(self) -> int:
        """
        Returns number of samples.

        Returns
        -------

        """
        return self._n

    @property
    def num_variables(self) -> int:
        """
        Returns number of samples.

        Returns
        -------

        """
        return self._p

    def p(self, metric="q2") -> float:
        """
        Calculates the significance of the constructed model by
        permutation test.

        Parameters
        ----------
        metric: str
            Metric used to assess the performance of the constructed
            model. "q2" and "error" are accepted as values.
            "q2": Q2
            "error": Misclassification error rate.

        Raises
        ------
        ValueError

        Returns
        -------
        float
            p value

        """
        if self._perm_err is None:
            raise ValueError("Permutation test has not been performed.")
        if metric not in ("q2", "error"):
            raise ValueError("Expected `q2`, `error`, got {}.".format(metric))

        if metric == "q2":
            nb: int = np.count_nonzero(self._perm_q2 >= self.q2) + 1
            nt: float = self._perm_q2.size + 1.
        else:
            err: float = self.min_nmc / self.y.size
            nb: int = np.count_nonzero(self._perm_err <= err) + 1
            nt: float = self._perm_err.size + 1.

        return nb / nt

    @staticmethod
    def _check_x(x) -> np.ndarray:
        """
        Checks the valid variables.

        Parameters
        ----------
        x: np.ndarray
            Data matrix

        Returns
        -------
        x: np.ndarray
            Valid data matrix with no NaNs and Infs.

        """
        # check nan and inf
        x[np.isnan(x)] = 0.
        x[np.isinf(x)] = 0.
        return x

    def _create_optimal_model(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Create final model based on the optimal number of components.
        """
        x = self._check_x(x.copy())

        # scale data matrix
        x_scale = self.scaler.fit(x, y)

        # optimal component number
        npc = self._opt_component + 1

        # fit the model
        self.estimator.fit(x_scale.copy(), y.copy(), npc)

        # calculate VIPs
        self.estimator.calculate_vip(npc)
        self._vip = self.estimator.vip

        # coefficients of centered and scaled X
        estimator, _ = self._create_scaler_estimator()
        scaler = pretreatment.Scaler(scaler="uv", fit_type="balanced")
        ys = (y - y.mean()) / y.std()
        x_uv = scaler.fit(x[:, self.scaler.variable_index], ys)
        estimator.fit(x_uv.copy(), ys.copy(), npc)
        self._coefcs = estimator.coefs[self._opt_component]

    def _reset_y(self, y) -> np.ndarray:
        """
        Reset the labels in y to 0 and 1, and name the groups using the
        labels in y.

        Parameters
        ----------
        y: np.ndarray | list

        Returns
        -------
        np.ndarray
            Label reset in y.

        """
        if isinstance(y, list):
            y = np.array([str(v) for v in y], dtype=str)

        # groups
        labels = np.unique(y)
        # only binary classification is allowed.
        if labels.size != 2:
            raise ValueError(
                "Only binary classification is currently accepted."
            )

        # reset the values for each class
        groups = collections.defaultdict()
        y_reset = np.zeros_like(y, dtype=np.float64)
        for i, label in enumerate(labels):
            y_reset[y == label] = i
            groups[i] = label if isinstance(label, str) else str(int(label))

        y_reset[y_reset == 0.] = -1.
        groups[-1] = groups[0]

        self.groups = groups
        return y_reset

    def _create_scaler_estimator(self):
        """
        Creates scaler and estimator.

        Returns
        -------

        """
        if not self.use_opls:
            return PLS(), pretreatment.Scaler(scaler=self._scaler_param)
        if self.use_opls:
            return OPLS(), pretreatment.Scaler(scaler=self._scaler_param)
