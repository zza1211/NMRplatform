"""
Orthogonal Projection on Latent Structure (O-PLS)
"""
import numpy as np
from typing import Optional, Tuple,Union
from .core import (correct_fit, correct_x_1d, correct_x_2d,
                   summary_opls, opls_vip)

class OPLS:
    """
    Orthogonal Projection on Latent Structure (O-PLS).
    Methods
    ----------
    predictive_scores: np.ndarray
        First predictive score.
    predictive_loadings: np.ndarray
        Predictive loadings.
    weights_y: np.ndarray
        y weights.
    orthogonal_loadings: np.ndarray
        Orthogonal loadings.
    orthogonal_scores: np.ndarray
        Orthogonal scores.

    Parameters
    ----------
    tol: float
        The tolerance for convergence in NIPALS.
    max_iter: int
        The maximum number of iterations.

    """
    def __init__(self, tol: float = 1e-10, max_iter: int = 1000):
        """
        TODO:
            1. add arg for specifying the method for performing PLS

        """
        # orthogonal score matrix
        self._Tortho: Optional[np.ndarray] = None
        # orthogonal loadings
        self._Portho: Optional[np.ndarray] = None
        # loadings
        self._Wortho: Optional[np.ndarray] = None
        # covariate weights
        self._w: Optional[np.ndarray] = None

        # predictive scores
        self._T: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._C: Optional[np.ndarray] = None
        self._W: Optional[np.ndarray] = None
        self._vip4o: Optional[np.ndarray] = None
        self._vip4p: Optional[np.ndarray] = None
        self._vip4t: Optional[np.ndarray] = None
        # coefficients
        self.coefs: Optional[np.ndarray] = None
        # statistics
        self.cov: Optional[np.ndarray] = None
        self.corr: Optional[np.ndarray] = None
        self.r2x: Optional[np.ndarray] = None
        self.r2x_cum: Optional[np.ndarray] = None
        self.r2y: Optional[np.ndarray] = None
        self.r2y_cum: Optional[np.ndarray] = None
        # total number of components
        self.npc: Optional[int] = None
        self.r2: Optional[float] = None
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, x, y, n_comp=None) -> None:
        """
        Fits an OPLS model.

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by p variables.
        y: np.ndarray
            Dependent matrix with size n samples by 1, or a vector
        n_comp: int
            Number of components, default is None, which indicates that
            the largest dimension which is the smaller value between n
            and p will be used.

        Returns
        -------
        OPLS object

        Reference
        ---------
        [1] Trygg J, Wold S. Projection on Latent Structure (OPLS).
            J Chemometrics. 2002, 16, 119-128.
        [2] Trygg J, Wold S. O2-PLS, a two-block (X-Y) latent variable
            regression (LVR) method with a integral OSC filter.
            J Chemometrics. 2003, 17, 53-64.

        """
        n, p = x.shape
        npc = min(n, p)
        if n_comp is not None and n_comp < npc:
            npc = n_comp

        yp, t_o, p_o, w_o, t_p, w_p, p_p, coefs, w_y, tw = correct_fit(
            x.copy(), y, npc, self.tol, self.max_iter)

        # get the statistics
        cov, corr, r2x, r2y = summary_opls(
            x, y, t_p, t_o, p_p, p_o, w_y, n_comp)

        self._Tortho = t_o
        self._Portho = p_o
        self._Wortho = w_o
        # covariate weights
        self._w = tw

        # predictive weights
        self._W = w_p
        # coefficients and predictive scores
        self._T = t_p
        self._P = p_p
        self._C = w_y
        self.coefs = coefs

        # summaries of the fits
        self.cov = cov
        self.corr = corr
        self.r2x = r2x
        self.r2x_cum = r2x
        self.r2y = r2y
        self.r2y_cum = r2y
        self.r2 = 1. - ((yp - y) ** 2).sum() / ((yp - yp.mean()) ** 2).sum()

        self.npc = npc

    def predict(self, x, n_component=None, return_scores=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict the new coming data matrix.
        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by p variables.
        n_component: int | None
            Number of components.
        return_scores: bool
            Whether the scores should be returned.

        Returns
        -------
        y: np.ndarray
            Predicted scores for classification.
        score: np.ndarray
            Predictive scores.
        """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        coef = self.coefs[n_component - 1]

        y = np.dot(x, coef)
        if return_scores:
            return y, np.dot(x, self._w.T)

        return y

    def correct(self, x, n_component=None, return_scores=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Correction of X

        Parameters
        ----------
        x: np.ndarray
            Data matrix with size n by c, where n is number of
            samples, and c is number of variables
        n_component: int | None
            Number of components. If is None, the number of components
            used in fitting the model is used. Default is None.
        return_scores: bool
            Return orthogonal scores. Default is False.

        Returns
        -------
        xc: np.ndarray
            Corrected data, with same matrix size with input X.
        t: np.ndarray
            Orthogonal score, n by n_component.

        """
        # TODO: Check X type and dimension consistencies between X and
        #       scores in model.
        if n_component is None:
            n_component = self.npc

        if x.ndim == 1:
            xc, t = correct_x_1d(x.copy(), self._Wortho[:n_component],
                                 self._Portho[:n_component])
        else:
            xc, t = correct_x_2d(x.copy(), self._Wortho[:n_component],
                                 self._Portho[:n_component])

        if return_scores:
            return xc, t

        return xc

    def predictive_score(self, n_component=None) -> np.ndarray:
        """
        Parameters
        ----------
        n_component: int
            The component number.

        Returns
        -------
        np.ndarray
            The first predictive score.

        """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        return self._T[n_component-1]

    def ortho_score(self, n_component=None) -> np.ndarray:
        """

        Parameters
        ----------
        n_component: int
            The component number.

        Returns
        -------
        np.ndarray
            The first orthogonal score.

        """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        return self._Tortho[n_component-1]

    def calculate_vip(self, num_comp: int) -> None:
        """
        Calculates VIPs

        Parameters
        ----------
        num_comp: Number of components.

        Returns
        -------
        np.ndarray
            VIP4,O: Orthogonal VIP.
        np.ndarray
            VIP4,P: Predictive VIP.
        np.ndarray
            VIP4,tot: Total VIP.

        References
        ----------
        [1] B. Galindo-Prieto, et al. Variable influence on projection
            (VIP) for orthogonal projections to latent structures
            (OPLS). J Chemometr. 2014, 28, 623-632.
        [2] B. Galindo-Prieto, et al. Variable influence on projection
            (VIP) for OPLS models and its applicability in multivariate
            time series analysis. Chem Int Lab Sys. 2015, 146, 297-304.

        Notes
        -----
            Here, only VIPs from orthonomal loadings P were
            implemented, i.e., VIP4 of ref, which is recommended.

        """
        if num_comp > self.npc:
            raise ValueError("The number of components input must not be "
                             "larger than the maximum number of "
                             f"components {self.npc}.")

        npc: int = num_comp - 1
        vip_o, vip_p, vip_t = opls_vip(self._Tortho, self._Portho,
                                       self._T[npc], self._P[npc])
        self._vip4o = vip_o
        self._vip4p = vip_p
        self._vip4t = vip_t

    @property
    def predictive_scores(self) -> np.ndarray:
        """ Orthogonal loadings. """
        return self._T.T

    @property
    def predictive_loadings(self) -> np.ndarray:
        """ Predictive loadings. """
        return self._P.T

    @property
    def weights_y(self) -> np.ndarray:
        """ y scores. """
        return self._C

    @property
    def orthogonal_loadings(self) -> np.ndarray:
        """ Orthogonal loadings. """
        return self._Portho.T

    @property
    def orthogonal_scores(self) -> np.ndarray:
        """ Orthogonal scores. """
        return self._Tortho.T

    @property
    def vip(self) -> np.ndarray:
        """
        Variable influence on projection for OPLS. This returns the
        total VIP4 of ref [1].

        References
        ----------
        [1] Galindo-Prieto B, et al. Variable influence on projection
            (VIP) for orthogonal projections to latent structures
            (OPLS). J Chemometrics. 2014, 28(8), 623-632.

        """
        return self._vip4t
