import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.fftpack import ifft
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from functools import partial

#from dynamics import dynamics


class Heston:
    def __init__(self, mu=np.nan, kappa=np.nan, theta=np.nan, sigma=np.nan, rho=np.nan):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.cov = np.array([[1, self.rho], [self.rho, 1]])

    @staticmethod
    def param_names():
        return ["mu", "kappa", "theta", "sigma", "rho"]

    @staticmethod
    def visible_state_names():
        return ["s"]

    @staticmethod
    def hidden_state_names():
        return ["nu"]

    @staticmethod
    def exog_names():
        return ["r"]

    @staticmethod
    def num_params():
        return 5

    @staticmethod
    def num_visible_states():
        return 1

    @staticmethod
    def num_hidden_states():
        return 1

    @staticmethod
    def num_exogs():
        return 1

    def set_params(self, p: list):
        self.mu = p[0]
        self.kappa = p[1]
        self.theta = p[2]
        self.sigma = p[3]
        self.rho = p[4]
        self.cov = np.array([[1, self.rho], [self.rho, 1]])

    def params(self):
        return [self.mu, self.kappa, self.theta, self.sigma, self.rho]

    @staticmethod
    def constraint(p):
        return (
            2 * p[1] * p[2] - np.square(p[3]),
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
        )  # first constraint: 2 * kappa * theta > sigma^2

    @staticmethod
    def cons_lb():
        return 0, 0, 0, 0, 0, -1, 0

    @staticmethod
    def cons_ub():
        return np.inf, 1, 1000, 1, 100, 1, 100

    @staticmethod
    def bounds():
        return ((0, 1), (0, 1000), (0, 1), (0, 100), (-1, 1), (0, 100))

    @staticmethod
    def example_paramset():
        return (0.01, 0.5, 0.1, 0.1, 0.6, 0.01)

    def simulate(self, s0, nu0, T, trials, dt):
        periods = int(np.ceil(T / dt))
        Z = np.random.multivariate_normal(np.zeros(2), self.cov, size=(trials, periods))
        nu = self.simulate_volatility(nu0, dt, Z)
        s = self.simulate_price(s0, nu, dt, Z)

        return s.T, nu.T

    def simulate_volatility(self, nu0, dt, Z):
        nu = np.zeros((Z.shape[0], Z.shape[1] + 1))
        nu[:, 0] = nu0

        for i in range(1, nu.shape[1]):
            nu[:, i] = (
                nu[:, i - 1]
                + self.kappa * (self.theta - nu[:, i - 1]) * dt
                + self.sigma * np.sqrt(nu[:, i - 1]) * Z[:, i - 1, 0] * np.sqrt(dt)
            )
            nu[:, i] = np.abs(nu[:, i])  # reflect

        return nu

    def simulate_price(self, s0, nu, dt, Z):
        s = np.zeros(nu.shape)
        s[:, 0] = s0

        for i in range(1, s.shape[1]):
            s[:, i] = (
                s[:, i - 1]
                + self.mu * s[:, i - 1] * dt
                + np.sqrt(nu[:, i]) * s[:, i - 1] * Z[:, i - 1, 1] * np.sqrt(dt)
            )
            s[:, i] = np.abs(s[:, i])

        return s

    def simulate_step(self, s0, nu0, dt):
        s, nu = self.simulate(s0, nu0, 1, 1, dt)
        return s[-1, 0], nu[-1, 0]

    def simulate_n(self, s0, nu0, n, dt):
        s, nu = self.simulate(s0, nu0, n, 1, dt)
        return s[-1, 0], nu[-1, 0]

    def price_call(self, s0, nu0, strike, T, r):
        cf = partial(self.characteristic, t=T, nu0=nu0)

        def Q1(k):
            integrand = lambda u: np.real(
                (np.exp(-u * k * 1j) / (u * 1j)) * cf(u - 1j) / cf(-1.0000000000001j)
            )
            return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, 1e3, limit=2000)[0]

        def Q2(k):
            integrand = lambda u: np.real(np.exp(-u * k * 1j) / (u * 1j) * cf(u))
            return 1 / 2 + 1 / np.pi * quad(integrand, 1e-15, 1e3, limit=2000)[0]

        k = np.log(strike / s0)
        return s0 * Q1(k) - strike * np.exp(-r * T) * Q2(k)

    def price_put(self, s0, nu0, strike, T, r):
        return self.price_call(s0, nu0, strike, T, r) - s0 + strike * np.exp(-r * T)

    def characteristic(self, u, t, nu0):
        """
        Heston characteristic function as proposed by Schoutens (2004)
        """
        v0, mu, kappa, theta, sigma, rho = (
            nu0,
            self.mu,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
        )
        xi = kappa - sigma * rho * u * 1j
        d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
        g1 = (xi + d) / (xi - d)
        g2 = 1 / g1
        cf = np.exp(
            1j * u * mu * t
            + (kappa * theta)
            / (sigma**2)
            * ((xi - d) * t - 2 * np.log((1 - g2 * np.exp(-d * t)) / (1 - g2)))
            + (v0 / sigma**2)
            * (xi - d)
            * (1 - np.exp(-d * t))
            / (1 - g2 * np.exp(-d * t))
        )
        return cf

    def price_call_batch(self, s0, nu0, strikes, T, r):
        cf = partial(self.characteristic, t=T, nu0=nu0)
        K = strikes

        N = 2**15  # FFT more efficient for N power of 2
        B = 500  # integration limit
        dx = B / N
        x = np.arange(N) * dx  # the final value B is excluded

        weight = np.arange(N)  # Simpson weights
        weight = 3 + (-1) ** (weight + 1)
        weight[0] = 1
        weight[N - 1] = 1

        dk = 2 * np.pi / B
        b = N * dk / 2
        ks = -b + dk * np.arange(N)

        integrand = (
            np.exp(-1j * b * np.arange(N) * dx)
            * cf(x - 0.5j)
            * 1
            / (x**2 + 0.25)
            * weight
            * dx
            / 3
        )
        integral_value = np.real(ifft(integrand) * N)

        # if interp == "linear":
        # 	spline_lin = interp1d(ks, integral_value, kind='linear')
        # 	prices = s0 - np.sqrt(s0 * K) * np.exp(-r*T)/np.pi * spline_lin( np.log(s0/K) )
        # elif interp == "cubic":
        spline_cub = interp1d(ks, integral_value, kind="cubic")
        prices = s0 - np.sqrt(s0 * K) * np.exp(-r * T) / np.pi * spline_cub(
            np.log(s0 / K)
        )
        return prices

    def price_put_batch(self, s0, nu0, strikes, T, r):
        return (
            self.price_call_batch(s0, nu0, strikes, T, r)
            - s0
            + strikes * np.exp(-r * T)
        )


if __name__ == "__main__":
    # r = 0.05                                           # drift
    # rho = -0.8                                         # correlation coefficient
    # kappa = 3                                          # mean reversion coefficient
    # theta = 0.1                                        # long-term mean of the variance
    # sigma = 0.25                                       # (Vol of Vol) - Volatility of instantaneous variance
    # T = 15                                             # Terminal time
    # K = 45	                                           # Stike
    # v0 = 0.08                                          # spot variance
    # S0 = 45                                            # spot stock price

    r = 0.01  # drift
    rho = 0.15  # correlation coefficient
    kappa = 10  # mean reversion coefficient
    theta = 0.13  # long-term mean of the variance
    sigma = 5  # (Vol of Vol) - Volatility of instantaneous variance
    T = 1  # Terminal time
    K = 420  # Stike
    v0 = 0.2  # spot variance
    S0 = 423  # spot stock price
    model = Heston(r, kappa, theta, sigma, rho)
    paths, _ = model.simulate(S0, v0, T, 3000, 1 / 252)

    # plt.plot(paths, c="red", alpha=0.01)
    # plt.title("Simulated Paths")
    # plt.show()

    # plt.hist(paths[-1, :], bins=50)
    # plt.show()

    # print(model.simulate_step(S0, v0, 1/360))
    # print(model.simulate_n(S0, v0, 10, 1/360/10))

    # print(model.price_call(S0,K,r,T,v0),model.price_put(S0,K,r,T,v0))

    strikes = np.arange(400, 440)
    fcp = model.price_call_batch(S0, v0, strikes, T, r)
    fpp = model.price_put_batch(S0, v0, strikes, T, r)

    cp = []
    pp = []
    for k in strikes:
        cp.append(model.price_call(S0, v0, k, T, r))
        pp.append(model.price_put(S0, v0, k, T, r))

    cp = np.array(cp)
    pp = np.array(pp)

    ce = fcp - cp
    pe = fpp - pp

    # print("\nprice_call_batch\n", fcp)
    print("\nHeston_price_call\n", cp)
    # print("\ncall_estimation_error\n", ce)

    # print("\nprice_put_batch\n", fpp)
    print("\nHeston_price_put\n", pp)
    # print("\nput_estimation_error\n", pe)
