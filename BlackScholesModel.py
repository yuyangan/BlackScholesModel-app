import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 0) Utilities: Normal distribution
# ============================================================

class NormalDist:
    @staticmethod
    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def pdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ============================================================
# 1) Domain model (OOP)
# ============================================================

class OptionKind(str, Enum):
    CALL = "call"
    PUT = "put"

class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"

@dataclass(frozen=True)
class VanillaOption:
    strike: float
    maturity: float  # years
    kind: OptionKind
    style: ExerciseStyle

    def payoff(self, S: float) -> float:
        if self.kind == OptionKind.CALL:
            return max(S - self.strike, 0.0)
        return max(self.strike - S, 0.0)

@dataclass(frozen=True)
class MarketData:
    spot: float
    rate: float      # r
    dividend: float  # q
    vol: float       # sigma


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


# ============================================================
# 2) Black-Scholes (European) pricer + Greeks
# ============================================================

class BlackScholesPricer:
    def _d1_d2(self, opt: VanillaOption, mkt: MarketData) -> Tuple[float, float]:
        S, K, T = mkt.spot, opt.strike, opt.maturity
        r, q, sigma = mkt.rate, mkt.dividend, mkt.vol
        vsqrt = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrt
        d2 = d1 - vsqrt
        return d1, d2

    def price(self, opt: VanillaOption, mkt: MarketData) -> float:
        if opt.maturity <= 0:
            return opt.payoff(mkt.spot)
        if opt.style != ExerciseStyle.EUROPEAN:
            raise ValueError("Black-Scholes is for EUROPEAN options.")

        S, K, T = mkt.spot, opt.strike, opt.maturity
        r, q, sigma = mkt.rate, mkt.dividend, mkt.vol
        if S <= 0 or K <= 0:
            raise ValueError("Spot and strike must be > 0.")
        if sigma <= 0:
            # deterministic forward limit
            fwd = S * math.exp((r - q) * T)
            disc = math.exp(-r * T)
            if opt.kind == OptionKind.CALL:
                return disc * max(fwd - K, 0.0)
            else:
                return disc * max(K - fwd, 0.0)

        d1, d2 = self._d1_d2(opt, mkt)
        df_r = math.exp(-r * T)
        df_q = math.exp(-q * T)

        if opt.kind == OptionKind.CALL:
            return S * df_q * NormalDist.cdf(d1) - K * df_r * NormalDist.cdf(d2)
        else:
            return K * df_r * NormalDist.cdf(-d2) - S * df_q * NormalDist.cdf(-d1)

    def greeks(self, opt: VanillaOption, mkt: MarketData) -> Greeks:
        if opt.style != ExerciseStyle.EUROPEAN:
            raise ValueError("Black-Scholes Greeks are for EUROPEAN options.")

        S, K, T = mkt.spot, opt.strike, opt.maturity
        r, q, sigma = mkt.rate, mkt.dividend, mkt.vol

        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            # Expiry / degenerate protection
            return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)

        d1, d2 = self._d1_d2(opt, mkt)
        df_r = math.exp(-r * T)
        df_q = math.exp(-q * T)
        nd1 = NormalDist.pdf(d1)

        gamma = (df_q * nd1) / (S * sigma * math.sqrt(T))
        vega = S * df_q * nd1 * math.sqrt(T)  # per +1.00 vol

        if opt.kind == OptionKind.CALL:
            delta = df_q * NormalDist.cdf(d1)
            theta = -(S * sigma * df_q * nd1) / (2.0 * math.sqrt(T)) - r * K * df_r * NormalDist.cdf(d2) + q * S * df_q * NormalDist.cdf(d1)
            rho = K * T * df_r * NormalDist.cdf(d2)  # per +1.00 rate
        else:
            delta = df_q * (NormalDist.cdf(d1) - 1.0)
            theta = -(S * sigma * df_q * nd1) / (2.0 * math.sqrt(T)) + r * K * df_r * NormalDist.cdf(-d2) - q * S * df_q * NormalDist.cdf(-d1)
            rho = -K * T * df_r * NormalDist.cdf(-d2)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

# ============================================================
# 2.5) Binomial Tree (CRR) pricer + Greeks
# ============================================================

@dataclass(frozen=True)
class BinomialSettings:
    steps: int = 800
    bump_rel: float = 1e-4  # for vega/rho bumps


class BinomialTreePricer:
    """
    CRR binomial tree. Supports European + American (early exercise).
    - Price: backward induction
    - Greeks:
        * Delta/Gamma: from first/second layer prices
        * Theta: maturity bump by one dt (per year)
        * Vega/Rho: bump-and-reprice
    """

    def _params(self, opt: VanillaOption, mkt: MarketData, settings: BinomialSettings):
        S0, K, T = mkt.spot, opt.strike, opt.maturity
        r, q, sigma = mkt.rate, mkt.dividend, mkt.vol
        N = int(settings.steps)

        if T <= 0:
            raise ValueError("Maturity must be > 0 for binomial parameters.")
        if N < 2:
            raise ValueError("Binomial steps must be >= 2 for stable greeks.")
        if S0 <= 0 or K <= 0:
            raise ValueError("Spot and strike must be > 0.")

        dt = T / N

        if sigma <= 0:
            # deterministic limit handled in price()
            u = d = 1.0
            p = 0.5
        else:
            u = math.exp(sigma * math.sqrt(dt))
            d = 1.0 / u
            p = (math.exp((r - q) * dt) - d) / (u - d)
            # clamp to avoid UI crashes under extreme params
            p = min(1.0, max(0.0, p))

        disc = math.exp(-r * dt)
        return N, dt, u, d, p, disc

    def price(self, opt: VanillaOption, mkt: MarketData, settings: BinomialSettings) -> float:
        if opt.maturity <= 0:
            return opt.payoff(mkt.spot)

        S0, K, T = mkt.spot, opt.strike, opt.maturity
        r, q, sigma = mkt.rate, mkt.dividend, mkt.vol

        if sigma <= 0:
            # deterministic forward limit
            fwd = S0 * math.exp((r - q) * T)
            discT = math.exp(-r * T)
            if opt.kind == OptionKind.CALL:
                return discT * max(fwd - K, 0.0)
            else:
                return discT * max(K - fwd, 0.0)

        N, dt, u, d, p, disc = self._params(opt, mkt, settings)

        # terminal stock prices S_T(j) = S0 * u^j * d^(N-j)
        j = np.arange(N + 1)
        S_T = S0 * (u ** j) * (d ** (N - j))

        if opt.kind == OptionKind.CALL:
            V = np.maximum(S_T - K, 0.0)
        else:
            V = np.maximum(K - S_T, 0.0)

        # backward induction
        if opt.style == ExerciseStyle.EUROPEAN:
            for i in range(N - 1, -1, -1):
                V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
            return float(V[0])

        # American: early exercise each node
        for i in range(N - 1, -1, -1):
            V = disc * (p * V[1:] + (1.0 - p) * V[:-1])

            j = np.arange(i + 1)
            S_i = S0 * (u ** j) * (d ** (i - j))
            payoff = np.array([opt.payoff(float(s)) for s in S_i], dtype=float)

            V = np.maximum(V, payoff)

        return float(V[0])

    def greeks(self, opt: VanillaOption, mkt: MarketData, settings: BinomialSettings) -> Greeks:
        N = int(settings.steps)
        if opt.maturity <= 0 or N < 2:
            return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)

        S0, K, T = mkt.spot, opt.strike, opt.maturity
        r, q, sigma = mkt.rate, mkt.dividend, mkt.vol
        if S0 <= 0 or K <= 0 or sigma <= 0:
            return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)

        N, dt, u, d, p, disc = self._params(opt, mkt, settings)

        # base
        V0 = self.price(opt, mkt, settings)

        # ---- Delta from level 1 ----
        Su, Sd = S0 * u, S0 * d
        Vu = self.price(opt, replace(mkt, spot=Su), settings)
        Vd = self.price(opt, replace(mkt, spot=Sd), settings)
        delta = (Vu - Vd) / (Su - Sd)

        # ---- Gamma from level 2 ----
        Suu, Sud, Sdd = S0 * u * u, S0 * u * d, S0 * d * d
        Vuu = self.price(opt, replace(mkt, spot=Suu), settings)
        Vud = self.price(opt, replace(mkt, spot=Sud), settings)
        Vdd = self.price(opt, replace(mkt, spot=Sdd), settings)

        # second derivative approx
        gamma = (
            (Vuu - Vud) / (Suu - Sud) - (Vud - Vdd) / (Sud - Sdd)
        ) / ((Suu - Sdd) / 2.0)

        # ---- Theta: move forward in calendar time by dt => maturity decreases ----
        T2 = max(T - dt, 1e-12)
        V_shorter = self.price(replace(opt, maturity=T2), mkt, settings)
        theta = (V_shorter - V0) / dt  # per year

        # ---- Vega/Rho bumps ----
        bump = settings.bump_rel

        dsig = max(1e-6, bump * max(mkt.vol, 1.0))
        vega = (
            self.price(opt, replace(mkt, vol=mkt.vol + dsig), settings)
            - self.price(opt, replace(mkt, vol=max(mkt.vol - dsig, 1e-12)), settings)
        ) / (2.0 * dsig)

        dr = max(1e-6, bump)
        rho = (
            self.price(opt, replace(mkt, rate=mkt.rate + dr), settings)
            - self.price(opt, replace(mkt, rate=mkt.rate - dr), settings)
        ) / (2.0 * dr)

        return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))


def _price_greeks_curve_binom(opt: VanillaOption, base_mkt: MarketData, binom_settings: BinomialSettings, S_grid: np.ndarray) -> pd.DataFrame:
    bt = BinomialTreePricer()
    out = []
    for s in S_grid:
        m = replace(base_mkt, spot=float(s))
        p = bt.price(opt, m, binom_settings)
        g = bt.greeks(opt, m, binom_settings)
        out.append([float(s), p, g.delta, g.gamma, g.vega, g.theta, g.rho])
    return pd.DataFrame(out, columns=["Spot", "Price", "Delta", "Gamma", "Vega", "Theta", "Rho"])


# ============================================================
# 3) Finite Difference (Theta-scheme) + PSOR for American
# ============================================================

@dataclass(frozen=True)
class FDMSettings:
    n_space: int = 250
    n_time: int = 250
    theta: float = 0.5            # 0.5 = Crank-Nicolson
    rannacher_steps: int = 2      # first steps fully implicit (damps kink oscillations)
    smax_mult: float = 4.0        # Smax = max(smax_mult*K, smax_mult*S0) baseline
    smax_sigma_boost: float = 6.0 # extra exp( boost*sigma*sqrt(T) )
    use_psor: bool = True
    psor_omega: float = 1.4
    psor_tol: float = 1e-8
    psor_max_iter: int = 100_000
    bump_rel: float = 1e-4        # for vega/rho bumps


@dataclass
class FDMResult:
    S: np.ndarray               # (N+1,)
    tau: np.ndarray             # (M+1,) time-to-maturity grid (0..T)
    V: Optional[np.ndarray]     # (M+1, N+1) if stored
    V_final: np.ndarray         # (N+1,) at tau=T
    V_prev: np.ndarray          # (N+1,) at tau=T-dt
    dt: float
    dS: float
    smax: float


def _thomas_solve(lower, diag, upper, rhs) -> np.ndarray:
    """Tridiagonal solve (Thomas algorithm)."""
    n = len(diag)
    cprime = np.zeros(n-1, dtype=float)
    dprime = np.zeros(n, dtype=float)

    cprime[0] = upper[0] / diag[0]
    dprime[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i-1] * cprime[i-1]
        if i < n-1:
            cprime[i] = upper[i] / denom
        dprime[i] = (rhs[i] - lower[i-1] * dprime[i-1]) / denom

    x = np.zeros(n, dtype=float)
    x[-1] = dprime[-1]
    for i in range(n-2, -1, -1):
        x[i] = dprime[i] - cprime[i] * x[i+1]
    return x


def _psor_solve(lower, diag, upper, rhs, payoff, omega, tol, max_iter) -> np.ndarray:
    """PSOR for American LCP: solve Ax=rhs with x>=payoff."""
    n = len(diag)
    x = np.maximum(payoff, rhs / diag)  # initial guess

    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(n):
            s = 0.0
            if i > 0:
                s += lower[i-1] * x[i-1]   # new
            if i < n-1:
                s += upper[i] * x[i+1]     # old (since not updated yet in this sweep)

            x_old = x[i]
            x_gs = (rhs[i] - s) / diag[i]
            x_sor = x_old + omega * (x_gs - x_old)
            x_new = max(payoff[i], x_sor)
            x[i] = x_new

            max_diff = max(max_diff, abs(x_new - x_old))

        if max_diff < tol:
            break

    return x


def _boundary_values(opt: VanillaOption, mkt: MarketData, tau: float, smax: float) -> Tuple[float, float]:
    """Dirichlet boundaries at S=0 and S=Smax."""
    K, r, q = opt.strike, mkt.rate, mkt.dividend
    if opt.kind == OptionKind.CALL:
        V0 = 0.0
        euro_asym = smax * math.exp(-q * tau) - K * math.exp(-r * tau)
        if opt.style == ExerciseStyle.AMERICAN:
            Vmax = max(smax - K, euro_asym)
        else:
            Vmax = euro_asym
    else:  # PUT
        if opt.style == ExerciseStyle.AMERICAN:
            V0 = K
        else:
            V0 = K * math.exp(-r * tau)
        Vmax = 0.0
    return V0, Vmax


@st.cache_data(show_spinner=False)
def fdm_solve(opt: VanillaOption, mkt: MarketData, settings: FDMSettings, store_matrix: bool) -> FDMResult:
    if opt.maturity <= 0:
        S = np.array([0.0, mkt.spot, 2.0*mkt.spot], dtype=float)
        Vfinal = np.array([opt.payoff(s) for s in S], dtype=float)
        return FDMResult(S=S, tau=np.array([0.0, 0.0]), V=None, V_final=Vfinal, V_prev=Vfinal.copy(),
                         dt=0.0, dS=max(mkt.spot, 1.0), smax=2.0*mkt.spot)

    S0, K, T = mkt.spot, opt.strike, opt.maturity
    r, q, sigma = mkt.rate, mkt.dividend, max(mkt.vol, 1e-12)

    N = int(settings.n_space)
    M = int(settings.n_time)
    if N < 20 or M < 20:
        raise ValueError("Use at least n_space>=20 and n_time>=20 for stable results.")

    # Choose Smax
    smax_base = max(settings.smax_mult * K, settings.smax_mult * S0)
    smax_ln = S0 * math.exp(settings.smax_sigma_boost * sigma * math.sqrt(T))
    smax = max(smax_base, smax_ln, 1.1*K, 1.1*S0)

    dS = smax / N
    dt = T / M

    S = np.linspace(0.0, smax, N+1)
    tau = np.linspace(0.0, T, M+1)

    payoff = np.array([opt.payoff(s) for s in S], dtype=float)
    V = payoff.copy()  # terminal at tau=0

    Vmat = None
    if store_matrix:
        Vmat = np.zeros((M+1, N+1), dtype=float)
        Vmat[0, :] = V

    V_prev = None

    # Precompute i=1..N-1 (interior nodes)
    Si = S[1:-1]
    a = 0.5 * (sigma**2) * (Si**2) / (dS**2) - (r - q) * Si / (2.0*dS)
    b = -(sigma**2) * (Si**2) / (dS**2) - r
    c = 0.5 * (sigma**2) * (Si**2) / (dS**2) + (r - q) * Si / (2.0*dS)

    # Time stepping
    for n in range(M):
        tau_n = tau[n]
        tau_np1 = tau[n+1]

        V0_n, VN_n = _boundary_values(opt, mkt, tau_n, smax)
        V0_np1, VN_np1 = _boundary_values(opt, mkt, tau_np1, smax)

        V[0] = V0_n
        V[-1] = VN_n

        theta_n = 1.0 if n < settings.rannacher_steps else settings.theta

        # Build A = I - theta dt L
        diag = 1.0 - theta_n * dt * b
        lower = -theta_n * dt * a[1:]      # length N-2
        upper = -theta_n * dt * c[:-1]     # length N-2

        # Build rhs = (I + (1-theta) dt L) V^n
        lam = (1.0 - theta_n) * dt
        rhs = (1.0 + lam * b) * V[1:-1] + lam * (a * V[:-2] + c * V[2:])

        # Boundary adjustments for time level n+1
        rhs[0]  += theta_n * dt * a[0]  * V0_np1
        rhs[-1] += theta_n * dt * c[-1] * VN_np1

        if opt.style == ExerciseStyle.AMERICAN and settings.use_psor:
            x = _psor_solve(lower, diag, upper, rhs, payoff[1:-1],
                            settings.psor_omega, settings.psor_tol, settings.psor_max_iter)
        else:
            x = _thomas_solve(lower, diag, upper, rhs)
            if opt.style == ExerciseStyle.AMERICAN:
                x = np.maximum(x, payoff[1:-1])

        V[0] = V0_np1
        V[-1] = VN_np1
        V[1:-1] = x

        if store_matrix:
            Vmat[n+1, :] = V

        if n == M-2:
            V_prev = V.copy()

    if V_prev is None:
        V_prev = V.copy()

    return FDMResult(S=S, tau=tau, V=Vmat, V_final=V.copy(), V_prev=V_prev.copy(), dt=dt, dS=dS, smax=smax)


def _interp_1d(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    if x0 <= x[0]:
        return float(y[0])
    if x0 >= x[-1]:
        return float(y[-1])
    return float(np.interp(x0, x, y))


def fdm_price(opt: VanillaOption, mkt: MarketData, settings: FDMSettings) -> float:
    res = fdm_solve(opt, mkt, settings, store_matrix=False)
    return _interp_1d(res.S, res.V_final, mkt.spot)


def _grid_derivatives(S: np.ndarray, V: np.ndarray, V_prev: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
    """Compute delta, gamma, theta arrays on the spot grid."""
    N = len(S) - 1
    dS = S[1] - S[0]

    delta = np.zeros_like(V)
    gamma = np.zeros_like(V)
    theta = np.zeros_like(V)

    # central differences interior
    delta[1:N] = (V[2:] - V[:-2]) / (2.0*dS)
    gamma[1:N] = (V[2:] - 2.0*V[1:-1] + V[:-2]) / (dS*dS)

    # one-sided at boundaries (simple)
    delta[0] = (V[1] - V[0]) / dS
    delta[N] = (V[N] - V[N-1]) / dS
    gamma[0] = gamma[1]
    gamma[N] = gamma[N-1]

    if dt > 0:
        # Theta = dV/dt = -dV/dtau ≈ (V(tau=T-dt) - V(tau=T)) / dt
        theta[:] = (V_prev - V) / dt
    else:
        theta[:] = 0.0

    return {"delta": delta, "gamma": gamma, "theta": theta}


@st.cache_data(show_spinner=False)
def fdm_greeks(opt: VanillaOption, mkt: MarketData, settings: FDMSettings, compute_vega_rho: bool) -> Tuple[float, Greeks, Dict[str, np.ndarray], FDMResult]:
    """Return price, point-greeks at S0, grid greeks arrays, and the FDMResult (for plots)."""
    res = fdm_solve(opt, mkt, settings, store_matrix=True)
    S = res.S
    V = res.V_final

    base_price = _interp_1d(S, V, mkt.spot)

    deriv = _grid_derivatives(S, res.V_final, res.V_prev, res.dt)
    delta0 = _interp_1d(S, deriv["delta"], mkt.spot)
    gamma0 = _interp_1d(S, deriv["gamma"], mkt.spot)
    theta0 = _interp_1d(S, deriv["theta"], mkt.spot)

    vega0 = float("nan")
    rho0 = float("nan")
    vega_grid = np.full_like(S, np.nan, dtype=float)
    rho_grid = np.full_like(S, np.nan, dtype=float)

    if compute_vega_rho:
        bump = settings.bump_rel

        # Vega via sigma bump (two extra PDE solves)
        dsig = max(1e-6, bump * max(mkt.vol, 1.0))
        res_up = fdm_solve(opt, replace(mkt, vol=mkt.vol + dsig), settings, store_matrix=False)
        res_dn = fdm_solve(opt, replace(mkt, vol=max(mkt.vol - dsig, 1e-12)), settings, store_matrix=False)
        vega_grid = (res_up.V_final - res_dn.V_final) / (2.0*dsig)
        vega0 = _interp_1d(S, vega_grid, mkt.spot)

        # Rho via r bump (two extra PDE solves)
        dr = max(1e-6, bump)
        res_up = fdm_solve(opt, replace(mkt, rate=mkt.rate + dr), settings, store_matrix=False)
        res_dn = fdm_solve(opt, replace(mkt, rate=mkt.rate - dr), settings, store_matrix=False)
        rho_grid = (res_up.V_final - res_dn.V_final) / (2.0*dr)
        rho0 = _interp_1d(S, rho_grid, mkt.spot)

    greeks0 = Greeks(delta0, gamma0, vega0, theta0, rho0)
    grids = {"delta": deriv["delta"], "gamma": deriv["gamma"], "theta": deriv["theta"],
             "vega": vega_grid, "rho": rho_grid}

    return base_price, greeks0, grids, res


# ============================================================
# 4) Implied volatility (BS implied vol)
# ============================================================

def _bs_price_for_sigma(target_sigma: float, opt: VanillaOption, mkt: MarketData, bs: BlackScholesPricer) -> float:
    return bs.price(opt, replace(mkt, vol=target_sigma))


def implied_vol_bs(
    market_price: float,
    opt: VanillaOption,
    mkt: MarketData,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Standard BS implied vol (European-equivalent IV).
    Uses robust bracketed Newton / bisection hybrid.
    """
    bs = BlackScholesPricer()

    S, K, T = mkt.spot, opt.strike, opt.maturity
    r, q = mkt.rate, mkt.dividend
    if T <= 0:
        return float("nan")

    df_r = math.exp(-r*T)
    df_q = math.exp(-q*T)

    # No-arbitrage bounds for European option prices
    if opt.kind == OptionKind.CALL:
        lower = max(0.0, S*df_q - K*df_r)
        upper = S*df_q
    else:
        lower = max(0.0, K*df_r - S*df_q)
        upper = K*df_r

    if not (lower - 1e-10 <= market_price <= upper + 1e-10):
        return float("nan")

    lo, hi = 1e-8, 5.0
    plo = _bs_price_for_sigma(lo, opt, mkt, bs)
    phi = _bs_price_for_sigma(hi, opt, mkt, bs)

    # ensure bracket
    for _ in range(60):
        if (plo - market_price) * (phi - market_price) <= 0:
            break
        hi *= 1.5
        if hi > 20.0:
            return float("nan")
        phi = _bs_price_for_sigma(hi, opt, mkt, bs)

    sigma = max(1e-4, mkt.vol)
    sigma = min(max(sigma, lo), hi)

    for _ in range(max_iter):
        price = _bs_price_for_sigma(sigma, opt, mkt, bs)
        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # update bracket monotonicity
        if diff > 0:
            hi = sigma
        else:
            lo = sigma

        # Newton step (using BS vega) with fallback to bisection
        g = bs.greeks(opt, replace(mkt, vol=sigma))
        vega = g.vega
        if vega > 1e-12:
            newton = sigma - diff / vega
            if lo < newton < hi:
                sigma = newton
            else:
                sigma = 0.5 * (lo + hi)
        else:
            sigma = 0.5 * (lo + hi)

    return sigma


# ============================================================
# 5) Streamlit UI + plotting helpers
# ============================================================

def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.6f}"

def _make_summary_table(opt: VanillaOption, mkt: MarketData, model_name: str, extra: Dict[str, float]) -> pd.DataFrame:
    rows = [
        ("Model", model_name),
        ("Style", opt.style.value),
        ("Kind", opt.kind.value),
        ("Spot (S)", mkt.spot),
        ("Strike (K)", opt.strike),
        ("Maturity (T, years)", opt.maturity),
        ("Rate (r)", mkt.rate),
        ("Dividend (q)", mkt.dividend),
        ("Vol (sigma)", mkt.vol),
    ]
    for k, v in extra.items():
        rows.append((k, v))
    df = pd.DataFrame(rows, columns=["Field", "Value"])
    return df


def _price_greeks_curve_bs(opt: VanillaOption, base_mkt: MarketData, S_grid: np.ndarray) -> pd.DataFrame:
    bs = BlackScholesPricer()
    out = []
    for s in S_grid:
        m = replace(base_mkt, spot=float(s))
        p = bs.price(opt, m)
        g = bs.greeks(opt, m)
        out.append([s, p, g.delta, g.gamma, g.vega, g.theta, g.rho])
    return pd.DataFrame(out, columns=["Spot", "Price", "Delta", "Gamma", "Vega", "Theta", "Rho"])


def _price_greeks_curve_fdm(opt: VanillaOption, mkt: MarketData, settings: FDMSettings, S_grid: np.ndarray, compute_vega_rho: bool) -> pd.DataFrame:
    # Use ONE PDE solve and interpolate greeks arrays for the curve.
    price0, g0, grids, res = fdm_greeks(opt, mkt, settings, compute_vega_rho=compute_vega_rho)

    out = []
    for s in S_grid:
        s = float(s)
        price_s = _interp_1d(res.S, res.V_final, s)
        delta_s = _interp_1d(res.S, grids["delta"], s)
        gamma_s = _interp_1d(res.S, grids["gamma"], s)
        theta_s = _interp_1d(res.S, grids["theta"], s)
        vega_s = _interp_1d(res.S, grids["vega"], s) if compute_vega_rho else float("nan")
        rho_s = _interp_1d(res.S, grids["rho"], s) if compute_vega_rho else float("nan")
        out.append([s, price_s, delta_s, gamma_s, vega_s, theta_s, rho_s])

    return pd.DataFrame(out, columns=["Spot", "Price", "Delta", "Gamma", "Vega", "Theta", "Rho"])


def _exercise_boundary_from_matrix(opt: VanillaOption, S: np.ndarray, tau: np.ndarray, Vmat: np.ndarray, tol: float = 1e-6) -> pd.DataFrame:
    """Approximate early exercise boundary over time (only meaningful for American)."""
    K = opt.strike
    payoff = np.array([opt.payoff(s) for s in S], dtype=float)
    rows = []
    for n in range(len(tau)):
        V = Vmat[n, :]
        ex = np.abs(V - payoff) <= tol

        if opt.kind == OptionKind.PUT:
            # exercise region typically S <= boundary, so boundary = max S where ex==True
            idx = np.where(ex)[0]
            boundary = float(S[idx].max()) if len(idx) > 0 else float("nan")
        else:
            # call (with dividends) exercise region typically S >= boundary
            idx = np.where(ex)[0]
            boundary = float(S[idx].min()) if len(idx) > 0 else float("nan")

        rows.append([tau[n], boundary])

    return pd.DataFrame(rows, columns=["Tau (time-to-maturity)", "ExerciseBoundaryS"])


# ============================================================
# 6) Streamlit App
# ============================================================

st.set_page_config(page_title="Options Lab (BS + American FDM)", layout="wide")

st.title("📈 Options Lab — Black–Scholes (European) + Finite Difference (American) + Greeks + Implied Vol")

with st.sidebar:
    st.header("Inputs")

    style = st.selectbox("Exercise Style", [ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN], format_func=lambda x: x.value)
    kind = st.selectbox("Option Kind", [OptionKind.CALL, OptionKind.PUT], format_func=lambda x: x.value)

    spot = st.number_input("Spot (S)", min_value=0.0001, value=100.0, step=1.0)
    strike = st.number_input("Strike (K)", min_value=0.0001, value=100.0, step=1.0)
    maturity = st.number_input("Maturity (T in years)", min_value=0.00001, value=1.0, step=0.25, format="%.6f")

    rate = st.number_input("Rate r (cont.)", value=0.05, step=0.01, format="%.6f")
    dividend = st.number_input("Dividend q (cont.)", value=0.00, step=0.01, format="%.6f")
    vol = st.number_input("Volatility sigma", min_value=0.000001, value=0.20, step=0.01, format="%.6f")

    st.divider()

    if style == ExerciseStyle.EUROPEAN:
        model = st.selectbox("Model for Pricing", [
            "Black–Scholes (Analytic)",
            "Finite Difference (Theta Scheme)",
            "Binomial Tree (CRR)"
        ])
    else:
        model = st.selectbox("Model for Pricing", [
            "Finite Difference (Theta Scheme)",
            "Binomial Tree (CRR)"
        ])

    st.divider()

    with st.expander("FDM Settings (American / PDE)", expanded=(style == ExerciseStyle.AMERICAN)):
        n_space = st.slider("n_space (spot steps)", 50, 800, 250, 10)
        n_time = st.slider("n_time (time steps)", 50, 800, 250, 10)
        theta = st.slider("theta (0=explicit, 0.5=CN, 1=implicit)", 0.0, 1.0, 0.5, 0.05)
        rannacher = st.slider("Rannacher steps (implicit warm-up)", 0, 8, 2, 1)

        smax_mult = st.slider("Smax multiplier baseline", 2.0, 10.0, 4.0, 0.5)
        smax_sigma_boost = st.slider("Smax lognormal boost (sigma*sqrt(T) multiplier)", 2.0, 10.0, 6.0, 0.5)

        use_psor = st.checkbox("Use PSOR (American LCP)", value=True)
        omega = st.slider("PSOR omega", 1.0, 1.95, 1.4, 0.05)
        tol_psor = st.number_input("PSOR tolerance", value=1e-8, format="%.1e")
        maxit_psor = st.number_input("PSOR max iterations", value=100000, step=10000)

        bump_rel = st.number_input("Bump size (relative) for Vega/Rho", value=1e-4, format="%.1e")

    st.divider()

    with st.expander("Binomial Settings", expanded=False):
        binom_steps = st.slider("Binomial steps", 50, 5000, 800, 50)
        binom_bump_rel = st.number_input("Binomial bump size (relative) for Vega/Rho", value=1e-4, format="%.1e")

    with st.expander("Plot Ranges"):
        s_min_mult = st.slider("Spot range min (mult * S)", 0.1, 1.0, 0.5, 0.05)
        s_max_mult = st.slider("Spot range max (mult * S)", 1.0, 3.0, 1.5, 0.05)
        n_pts = st.slider("Number of curve points", 25, 400, 120, 5)

        vol_min = st.slider("Vol surface min", 0.01, 1.0, 0.05, 0.01)
        vol_max = st.slider("Vol surface max", 0.05, 2.0, 0.60, 0.05)
        vol_pts = st.slider("Vol surface points", 10, 80, 25, 1)

    compute_vega_rho_american = st.checkbox("Compute Vega & Rho for American (slower)", value=True)

opt = VanillaOption(strike=float(strike), maturity=float(maturity), kind=kind, style=style)
mkt = MarketData(spot=float(spot), rate=float(rate), dividend=float(dividend), vol=float(vol))

settings = FDMSettings(
    n_space=int(n_space),
    n_time=int(n_time),
    theta=float(theta),
    rannacher_steps=int(rannacher),
    smax_mult=float(smax_mult),
    smax_sigma_boost=float(smax_sigma_boost),
    use_psor=bool(use_psor),
    psor_omega=float(omega),
    psor_tol=float(tol_psor),
    psor_max_iter=int(maxit_psor),
    bump_rel=float(bump_rel),
)

binom_settings = BinomialSettings(
    steps=int(binom_steps),
    bump_rel=float(binom_bump_rel),
)
S_grid_curve = np.linspace(s_min_mult*mkt.spot, s_max_mult*mkt.spot, int(n_pts))


tabs = st.tabs(["Summary", "Greeks Curves", "Surfaces & Heatmaps", "Implied Volatility", "Model Details (Formulas)"])


# ============================================================
# TAB 1: Summary
# ============================================================

with tabs[0]:

    # -------------------------------
    # Pricing + Greeks (logic unchanged)
    # -------------------------------
    if style == ExerciseStyle.EUROPEAN and model == "Black–Scholes (Analytic)":
        bs = BlackScholesPricer()
        price0 = bs.price(opt, mkt)
        g0 = bs.greeks(opt, mkt)
        extra = {"Model Notes": "Analytic Black–Scholes"}
        model_name = "Black–Scholes (Analytic)"
        fdm_res = None
    else:
        price0, g0, grids, fdm_res = fdm_greeks(
            opt, mkt, settings,
            compute_vega_rho=compute_vega_rho_american
        )
        extra = {
            "Smax": float(fdm_res.smax),
            "dS": float(fdm_res.dS),
            "dt": float(fdm_res.dt),
            "theta": settings.theta,
            "rannacher_steps": settings.rannacher_steps,
            "PSOR": int(settings.use_psor),
        }
        model_name = "Finite Difference (Theta Scheme)"

    # ===============================
    # TOP ROW — FULL WIDTH GREEKS
    # ===============================
    st.subheader("Greeks")

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Delta", _fmt(g0.delta))
    g2.metric("Gamma", _fmt(g0.gamma))
    g3.metric("Vega",  _fmt(g0.vega))
    g4.metric("Theta", _fmt(g0.theta))
    g5.metric("Rho",   _fmt(g0.rho))

    st.caption(
        "Vega/Rho per +1.00 change (×0.01 for 1%). "
        "Theta per year (÷365 for per day)."
    )

    st.divider()

    # ===============================
    # BELOW — TWO COLUMN LAYOUT
    # ===============================
    colA, colB = st.columns([1.2, 1.0], gap="large")

    # -------- LEFT COLUMN --------
    with colA:
        st.subheader("Price")
        st.metric("Option Price", _fmt(price0))

        st.subheader("Inputs & Model Settings")
        df_summary = _make_summary_table(opt, mkt, model_name, extra)
        st.dataframe(df_summary, use_container_width=True)

    # -------- RIGHT COLUMN --------
    with colB:
        st.subheader("Payoff + Value vs Spot")

        payoff_vals = np.array(
            [opt.payoff(s) for s in S_grid_curve], dtype=float
        )

        if style == ExerciseStyle.EUROPEAN and model == "Black–Scholes (Analytic)":
            curve_df = _price_greeks_curve_bs(opt, mkt, S_grid_curve)
        elif model == "Binomial Tree (CRR)":
            curve_df = _price_greeks_curve_binom(opt, mkt, binom_settings, S_grid_curve)
        else:
            curve_df = _price_greeks_curve_fdm(
                opt, mkt, settings,
                S_grid_curve,
                compute_vega_rho_american
            )
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curve_df["Spot"],
            y=curve_df["Price"],
            mode="lines",
            name="Option Value"
        ))
        fig.add_trace(go.Scatter(
            x=curve_df["Spot"],
            y=payoff_vals,
            mode="lines",
            name="Payoff (Intrinsic)"
        ))

        fig.update_layout(
            height=420,
            xaxis_title="Spot",
            yaxis_title="Value"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Curve Table")
        st.dataframe(
            curve_df.round({
                "Price": 2,
                "Delta": 2,
                "Gamma": 2,
                "Vega": 2,
                "Theta": 2,
                "Rho": 2
            }),
            use_container_width=True
        )

        st.download_button(
            "Download Curve CSV",
            curve_df.to_csv(index=False).encode("utf-8"),
            file_name="curve.csv",
            mime="text/csv"
        )

# ============================================================
# TAB 2: Greeks Curves
# ============================================================

with tabs[1]:
    st.subheader("Greeks vs Spot (Curves + Table)")

    if style == ExerciseStyle.EUROPEAN and model == "Black–Scholes (Analytic)":
        df = _price_greeks_curve_bs(opt, mkt, S_grid_curve)
        method_note = "Computed using analytic Black–Scholes formulas at each spot."
    elif model == "Binomial Tree (CRR)":
        df = _price_greeks_curve_binom(opt, mkt, binom_settings, S_grid_curve)
        method_note = "Computed using CRR binomial tree at each spot (includes early exercise if American)."
    else:
        df = _price_greeks_curve_fdm(opt, mkt, settings, S_grid_curve, compute_vega_rho_american)
        method_note = "Computed from ONE PDE solve (plus bumps for Vega/Rho if enabled), then interpolated across spot."

    st.caption(method_note)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        fig = px.line(df, x="Spot", y="Delta", title="Delta vs Spot")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(df, x="Spot", y="Gamma", title="Gamma vs Spot")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(df, x="Spot", y="Theta", title="Theta vs Spot")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(df, x="Spot", y="Price", title="Price vs Spot")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(df, x="Spot", y="Vega", title="Vega vs Spot")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(df, x="Spot", y="Rho", title="Rho vs Spot")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Greeks Table")
    st.dataframe(df.round(6), use_container_width=True)
    st.download_button("Download Greeks Table CSV", df.to_csv(index=False).encode("utf-8"), "greeks_table.csv", "text/csv")


# ============================================================
# TAB 3: Surfaces & Heatmaps
# ============================================================

with tabs[2]:
    st.subheader("Surfaces & Heatmaps")

    if style == ExerciseStyle.EUROPEAN and model == "Black–Scholes (Analytic)":
        st.markdown("### Price Surface: Spot × Vol (Black–Scholes)")
        S_surf = np.linspace(s_min_mult*mkt.spot, s_max_mult*mkt.spot, 60)
        V_surf = np.linspace(vol_min, vol_max, int(vol_pts))

        bs = BlackScholesPricer()
        Z = np.zeros((len(V_surf), len(S_surf)), dtype=float)
        for i, sig in enumerate(V_surf):
            for j, s in enumerate(S_surf):
                Z[i, j] = bs.price(opt, replace(mkt, spot=float(s), vol=float(sig)))

        colA, colB = st.columns(2, gap="large")
        with colA:
            fig = px.imshow(Z, x=S_surf, y=V_surf, aspect="auto", origin="lower",
                            labels={"x": "Spot", "y": "Vol", "color": "Price"},
                            title="Heatmap: Price(Spot, Vol)")
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            fig = go.Figure(data=[go.Surface(x=S_surf, y=V_surf, z=Z)])
            fig.update_layout(height=520, scene=dict(xaxis_title="Spot", yaxis_title="Vol", zaxis_title="Price"),
                              title="3D Surface: Price(Spot, Vol)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Sensitivity Heatmap: Price vs (Spot × Rate)")
        r_grid = np.linspace(mkt.rate - 0.10, mkt.rate + 0.10, 25)
        Z2 = np.zeros((len(r_grid), len(S_surf)), dtype=float)
        for i, rr in enumerate(r_grid):
            for j, s in enumerate(S_surf):
                Z2[i, j] = bs.price(opt, replace(mkt, spot=float(s), rate=float(rr)))

        fig = px.imshow(Z2, x=S_surf, y=r_grid, aspect="auto", origin="lower",
                        labels={"x": "Spot", "y": "Rate", "color": "Price"},
                        title="Heatmap: Price(Spot, Rate)")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown("### PDE Heatmap: Value vs Spot and Time-to-Maturity (FDM)")

        # Use stored matrix from cached fdm_greeks call
        price0, g0, grids, res = fdm_greeks(opt, mkt, settings, compute_vega_rho=compute_vega_rho_american)

        if res.V is None:
            st.warning("Matrix storage disabled unexpectedly. Try re-run; it should be on in this tab.")
        else:
            # Heatmap: tau on y, spot on x
            fig = px.imshow(res.V, x=res.S, y=res.tau, origin="lower", aspect="auto",
                            labels={"x": "Spot", "y": "Tau (time-to-maturity)", "color": "Value"},
                            title="Heatmap: V(Spot, Tau)")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Early Exercise Boundary (Approx.)")
            if opt.style == ExerciseStyle.AMERICAN:
                boundary_df = _exercise_boundary_from_matrix(opt, res.S, res.tau, res.V, tol=1e-6)
                fig = px.line(boundary_df, x="Tau (time-to-maturity)", y="ExerciseBoundaryS",
                              title="Exercise Boundary vs Tau (Approx.)")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(boundary_df.round(6), use_container_width=True)
            else:
                st.info("Exercise boundary is only meaningful for American options.")

            st.markdown("### Final Slice (Today): V(S, tau=T)")
            df_slice = pd.DataFrame({"Spot": res.S, "Value": res.V_final})
            fig = px.line(df_slice, x="Spot", y="Value", title="Final Value Slice")
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 4: Implied Volatility
# ============================================================

with tabs[3]:
    st.subheader("Implied Volatility (BS Implied Vol)")

    st.markdown(
        """
**Important practical note**  
Implied volatility is usually quoted as **Black–Scholes implied vol** (European-equivalent), even for American options.
So below we compute IV by solving:  
\[
BSPrice(\sigma) = \text{Market Price}
\]
"""
    )

    col1, col2 = st.columns([1.0, 1.2], gap="large")

    with col1:
        st.markdown("### Single Implied Vol")
        market_price = st.number_input("Market option price", min_value=0.0, value=float(price0), step=0.1, format="%.6f")

        opt_euro_equiv = VanillaOption(strike=opt.strike, maturity=opt.maturity, kind=opt.kind, style=ExerciseStyle.EUROPEAN)
        iv = implied_vol_bs(market_price, opt_euro_equiv, mkt)

        st.metric("Implied Vol (sigma)", _fmt(iv))

        st.markdown("### IV Smile (table → compute → plot)")
        n_smile = st.slider("Number of strikes", 5, 41, 15, 2)

        Ks = np.linspace(0.7*mkt.spot, 1.3*mkt.spot, int(n_smile))
        bs = BlackScholesPricer()

        # default: generate synthetic "market prices" from BS using current sigma (user can edit)
        base_rows = []
        for K in Ks:
            o = VanillaOption(strike=float(K), maturity=opt.maturity, kind=opt.kind, style=ExerciseStyle.EUROPEAN)
            p = bs.price(o, mkt)
            base_rows.append([float(K), float(p)])

        base_df = pd.DataFrame(base_rows, columns=["Strike", "MarketPrice"])
        edited = st.data_editor(base_df, use_container_width=True, num_rows="fixed")

    with col2:
        smile = edited.copy()
        ivs = []
        for _, row in smile.iterrows():
            K = float(row["Strike"])
            mp = float(row["MarketPrice"])
            o = VanillaOption(strike=K, maturity=opt.maturity, kind=opt.kind, style=ExerciseStyle.EUROPEAN)
            ivs.append(implied_vol_bs(mp, o, mkt))
        smile["ImpliedVol"] = ivs
        smile["Moneyness (K/S)"] = smile["Strike"] / mkt.spot

        fig = px.line(smile, x="Strike", y="ImpliedVol", markers=True, title="Implied Vol Smile: IV vs Strike")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(smile, x="Moneyness (K/S)", y="ImpliedVol", markers=True, title="Implied Vol vs Moneyness (K/S)")
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(smile.round(6), use_container_width=True)
        st.download_button("Download IV Smile CSV", smile.to_csv(index=False).encode("utf-8"), "iv_smile.csv", "text/csv")


# ============================================================
# TAB 5: Model Details (Formulas)
# ============================================================

with tabs[4]:
    st.subheader("Formulas & Methodology (what you code is what you plot)")

    # --------------------------------------------------------
    # Black–Scholes
    # --------------------------------------------------------
    st.markdown("## Black–Scholes (European)")

    st.latex(r"""
    d_1 = \frac{\ln(S/K) + (r-q+\frac12\sigma^2)T}{\sigma\sqrt{T}}
    """)

    st.latex(r"""
    d_2 = d_1 - \sigma\sqrt{T}
    """)

    st.markdown("**Call price**")
    st.latex(r"""
    C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2)
    """)

    st.markdown("**Put price**")
    st.latex(r"""
    P = K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1)
    """)

    st.divider()

    # --------------------------------------------------------
    # Greeks
    # --------------------------------------------------------
    st.markdown("## Greeks (European, analytic)")

    st.markdown("**Delta**")
    st.latex(r"""
    \Delta_C = e^{-qT}N(d_1), \quad
    \Delta_P = e^{-qT}(N(d_1)-1)
    """)

    st.markdown("**Gamma**")
    st.latex(r"""
    \Gamma = \frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}
    """)

    st.markdown("**Vega**")
    st.latex(r"""
    \nu = S e^{-qT}\phi(d_1)\sqrt{T}
    """)

    st.markdown("**Theta** (finance convention: $\\Theta = \\partial V / \\partial t$)")
    st.latex(r"""
    \Theta_C =
    -\frac{S\sigma e^{-qT}\phi(d_1)}{2\sqrt{T}}
    - rK e^{-rT}N(d_2)
    + qS e^{-qT}N(d_1)
    """)

    st.latex(r"""
    \Theta_P =
    -\frac{S\sigma e^{-qT}\phi(d_1)}{2\sqrt{T}}
    + rK e^{-rT}N(-d_2)
    - qS e^{-qT}N(-d_1)
    """)

    st.markdown("**Rho**")
    st.latex(r"""
    \rho_C = KT e^{-rT}N(d_2), \quad
    \rho_P = -KT e^{-rT}N(-d_2)
    """)

    st.divider()

    # --------------------------------------------------------
    # Finite Difference PDE
    # --------------------------------------------------------
    st.markdown("## Finite Difference PDE (European / American)")

    st.markdown("Using time-to-maturity $\\tau = T - t$:")

    st.latex(r"""
    \frac{\partial V}{\partial \tau}
    =
    \frac12\sigma^2 S^2 V_{SS}
    + (r-q)S V_S
    - rV
    """)

    st.markdown("### Discretization grid")

    st.latex(r"""
    S_i = i\Delta S, \quad i = 0,\dots,N
    """)

    st.latex(r"""
    \tau_n = n\Delta\tau, \quad n = 0,\dots,M
    """)

    st.markdown("### Central differences")

    st.latex(r"""
    V_S \approx \frac{V_{i+1}-V_{i-1}}{2\Delta S}, \quad
    V_{SS} \approx \frac{V_{i+1}-2V_i+V_{i-1}}{(\Delta S)^2}
    """)

    st.markdown("### Operator coefficients")

    st.latex(r"""
    a_i = \frac12\sigma^2\frac{S_i^2}{(\Delta S)^2}
    - \frac{(r-q)S_i}{2\Delta S}
    """)

    st.latex(r"""
    b_i = -\sigma^2\frac{S_i^2}{(\Delta S)^2} - r
    """)

    st.latex(r"""
    c_i = \frac12\sigma^2\frac{S_i^2}{(\Delta S)^2}
    + \frac{(r-q)S_i}{2\Delta S}
    """)

    st.markdown("### Theta-scheme time stepping")

    st.latex(r"""
    (I-\theta\Delta\tau L)V^{n+1}
    =
    (I+(1-\theta)\Delta\tau L)V^n
    """)

    st.markdown("### American constraint (early exercise)")

    st.latex(r"""
    V_i^{n+1} \ge \Phi(S_i)
    """)

    st.markdown("This yields a Linear Complementarity Problem solved via **PSOR**:")

    st.latex(r"""
    x_i^{GS} =
    \frac{1}{d_i}
    \left(
    rhs_i - \ell_i x_{i-1} - u_i x_{i+1}
    \right)
    """)

    st.latex(r"""
    x_i^{SOR}
    =
    x_i^{old}
    + \omega\left(x_i^{GS}-x_i^{old}\right)
    """)

    st.latex(r"""
    x_i^{new} = \max(\Phi_i,\;x_i^{SOR})
    """)

    st.markdown("### Greeks from the grid (American)")

    st.markdown("""
    - Delta / Gamma: finite differences on the final time slice  
    - Theta: backward difference in time  
    - Vega / Rho: bump-and-reprice (re-solve the PDE)
    """)

    st.latex(r"""
    \Theta \approx
    \frac{V(\tau=T-\Delta\tau)-V(\tau=T)}{\Delta\tau}
    """)


