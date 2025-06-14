import numpy as np

from typing import Optional

from .core import pls_c, pls_vip, summary_pls


class PLS:
    """ Partial least squares. """
    def __init__(self):
        self._T: Optional[np.ndarry] = None
        self._P: Optional[np.ndarry] = None
        self._W: Optional[np.ndarry] = None
        self._C: Optional[np.ndarry] = None
        self.coefs: Optional[np.ndarry] = None
        self._vips: Optional[np.ndarray] = None

        self.r2x: Optional[np.ndarray] = None
        self.r2x_cum: Optional[np.ndarray] = None
        self.r2y: Optional[np.ndarray] = None
        self.r2y_cum: Optional[np.ndarray] = None
        self.r2: Optional[float] = None

    def fit(self, x: np.ndarray, y: np.ndarray, num_comp: int) -> None:
        """
        Fit PLS model

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n by p, where n number
            of samples/instances, p number of variables
        y: np.ndarray
            Dependent variable with size n by 1
        num_comp: int
            Number of components. Default is None, which indicates that
            smaller number between n and p will be used.

        Returns
        -------
        PLS object

        """
        n, p = x.shape
        if num_comp > min(n, p):
            raise ValueError(f"Number of components {num_comp} exceeds the "
                             f"number of samples {n} or variables {p}.")
        #TODO 搞清楚为啥
        # if num_comp == 1:
        #     num_comp = 2
        t, w, p, c, coefs = pls_c(x.copy(), y.copy(), num_comp)

        r2x, r2x_cum, r2y, r2y_cum = summary_pls(x, y, t, p, c, num_comp)

        # save results to matrix
        self._T = t
        self._P = p
        self._W = w
        self._C = c
        self.coefs = coefs

        self.r2x = r2x
        self.r2y = r2y
        self.r2x_cum = r2x_cum
        self.r2y_cum = r2y_cum

        # calculate R2
        yp = np.dot(x, coefs[num_comp - 1])
        self.r2 = 1. - ((yp - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    def predict(self, x, n_component=None) -> np.ndarray:
        """
        Predicts the input data matrix.

        Parameters
        ----------
        x: np.ndarray
            x for prediction
        n_component: int
            Number of components. Defaults to None, which indicates that
            the number of components previously set will be used.

        Returns
        -------
        np.ndarray

        """
        npc: int = self.coefs.shape[0]
        if n_component is not None and n_component > npc:
            raise ValueError(f"Number of components {n_component} exceeds "
                             f"the determined number of components {npc}.")
        if n_component is None:
            npc -= 1
        else:
            npc = n_component - 1

        return np.dot(x, self.coefs[npc])

    def calculate_vip(self, num_comp: int) -> None:
        """
        Calculates variable importance in projection.

        Returns
        -------
        np.ndarray
            Variable importance in projection.

        """
        npc: int = self._T.shape[0]
        if num_comp > npc:
            raise ValueError("The number of components input must not be "
                             "larger than the maximum number of "
                             f"components {npc}.")

        self._vips = pls_vip(self._W[:num_comp], self._T[:num_comp],
                             self._C[:num_comp])

    @property
    def vip(self) -> np.ndarray:
        """
        Variable importance in projection.

        Returns
        -------

        """
        return self._vips[-1]

    @property
    def scores_x(self) -> np.ndarray:
        """
        Scores.

        Returns
        -------
        np.ndarray
            X Scores

        """
        return self._T.T

    @property
    def loadings_x(self) -> np.ndarray:
        """
        Loadings.

        Returns
        -------
        np.ndarray
            loadings

        """
        return self._P.T

    @property
    def weights_y(self) -> np.ndarray:
        """
        y weights.

        Returns
        -------
        np.ndarray
            y scores

        """
        return self._C

    @property
    def weigths_x(self):
        """

        Returns
        -------
            np.ndarray
                x weights

        """
        return self._W
