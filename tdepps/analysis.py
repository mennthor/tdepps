import numpy as np
import scipy.optimize as sco


class TransientsAnalysis(object):
    def __init__(self, X, llh, bg_inj, bg_rate_inj):
        self.X = X
        self.llh = llh
        self.bg_inj = bg_inj
        self.bg_rate_inj = bg_rate_inj
        return

    def do_trials(self, n_trials, signal_inj=None):
        if signal_inj is not None:
            raise NotImplementedError("Signal injection not yet implemented.")

        res = np.empty(n_trials, dtype=np.float)
        for i in range(n_trials):
            res[i] = self.fit_lnllh_ratio_params(X, theta0, args, **kwargs)


        return

    def fit_lnllh_ratio_params(self, X, theta0, args, **kwargs):
        def _llh(x):
            theta = {"ns": x}
            func, grad = self.llh.lnllh_ratio(X, theta, args)
            return -1. * func, -1. * grad

        bounds = kwargs.pop("bounds", [[0, len(self.X)]])

        res = sco.minimize(fun=_llh, x0=theta0, jac=True,
                           bounds=bounds, **kwargs)

        return res.x
