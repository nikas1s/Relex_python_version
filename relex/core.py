"""Core solver for the RELEX relativistic Coulomb-excitation model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import CubicSpline


@dataclass
class RelexResult:
    inputs: dict
    b: np.ndarray
    spinave: np.ndarray  # shape (nst, nb)
    cross_sections: dict[int, float]
    stat_by_b: list[np.ndarray]  # list of (nst, 5, 9) complex arrays


# -----------------------------
# Angular-momentum and factorial utilities
# -----------------------------

def gfv(max_n: int = 130) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sign and factorial arrays (iv, fak, fad)."""
    if max_n > 130:
        raise ValueError("max_n must be <= 130")
    iv = np.zeros(max_n + 1, dtype=float)
    fak = np.zeros(max_n + 1, dtype=float)
    fad = np.zeros(max_n + 1, dtype=float)
    iv[0] = 1.0
    fak[0] = 1.0
    fad[0] = 1.0
    if max_n >= 1:
        fad[1] = 1.0
    for i in range(1, max_n + 1):
        iv[i] = -iv[i - 1]
        fak[i] = i * fak[i - 1]
    for i in range(3, max_n + 1, 2):
        fad[i] = i * fad[i - 2]
    for i in range(2, max_n + 1, 2):
        fad[i] = i * fad[i - 2]
    return iv, fak, fad


_threej_f = None


def threej(a1: float, a2: float, a3: float, b1: float, b2: float, b3: float) -> float:
    """Compute the Wigner 3-j coefficient used in RELEX coupling matrix elements."""
    global _threej_f
    if _threej_f is None:
        f = [0.0] * 85
        f[0] = 1.0
        for i in range(1, 85):
            f[i] = f[i - 1] * float(i) / 15.0
        _threej_f = f
    f = _threej_f

    # Convert to float
    a1 = float(a1)
    a2 = float(a2)
    a3 = float(a3)
    b1 = float(b1)
    b2 = float(b2)
    b3 = float(b3)

    # The internal Clebsch-Gordan summation uses the opposite third-projection sign.
    b3 = -b3

    clebsh = 0.0
    three = 0.0
    if abs(b1) - a1 > 0.01:
        return 0.0
    if abs(b2) - a2 > 0.01:
        return 0.0
    if abs(b3) - a3 > 0.01:
        return 0.0
    if abs(b1 + b2 - b3) > 0.01:
        return 0.0
    if a3 - a1 - a2 > 0.01:
        return 0.0
    if abs(a1 - a2) - a3 > 0.01:
        return 0.0

    l1 = int(a1 + a2 - a3 + 1.01)
    l2 = int(a1 + b1 + 1.01)
    l3 = int(a2 + b2 + 1.01)
    l4 = int(a2 - b2 + 1.01)
    l5 = int(a3 + a2 - a1 + 1.01)
    l6 = int(a1 - b1 + 1.01)
    l7 = int(a3 + b3 + 1.01)
    l8 = int(a3 - b3 + 1.01)
    l9 = int(a3 + a1 - a2 + 1.01)
    l10 = int(a3 + a1 + a2 + 1.01)
    m1 = l1
    m2 = l6
    m3 = l3
    m4 = int((l7 + l8 - l3 - l4 + l2 - l6) / 2 + 1)
    m5 = int((l7 + l8 - l2 - l6 - l3 + l4) / 2 + 1)
    m6 = 1

    zw = math.sqrt(f[l1 - 1] * f[l2 - 1] * f[l6 - 1])
    tw = math.sqrt(f[l9 - 1] / f[l10 - 1] * (2.0 * a3 + 1.0) / float(l10))
    sw = tw * math.sqrt(f[l3 - 1] * f[l7 - 1] * f[l8 - 1] * f[l4 - 1] * f[l5 - 1])

    min_m = max(-m4, -m5, -m6) + 2
    max_m = min(m1, m2, m3)
    fr = -(-1.0) ** min_m
    if max_m < min_m:
        return 0.0
    for m in range(min_m, max_m + 1):
        mw = m - 1
        clebsh = clebsh + sw * fr / f[m4 + mw - 1] / (f[m2 - mw - 1] / zw) / f[m3 - mw - 1] / f[m1 - mw - 1] / f[m5 + mw - 1] / f[m6 + mw - 1]
        fr = -fr
    three = ((-1) ** int(round(a1 - a2 + b3))) / math.sqrt(2.0 * a3 + 1.0) * clebsh

    return three


@dataclass
class Context:
    beta: float
    gamma: float
    ei: complex
    pi: float
    hc: float
    xi: np.ndarray
    mm: np.ndarray
    n: int
    psitot: np.ndarray
    psinuc: np.ndarray
    bb: float
    iopnuc: int
    delr: float
    ngrid: int
    iv: np.ndarray
    fak: np.ndarray


# -----------------------------
# Physics helper functions
# -----------------------------

def rhopp(rg: float, ap: float, zp: float) -> float:
    pi = 3.141597265
    tpp = 2.4
    bpp = 0.413 * tpp
    deltap = ((ap - 2.0 * zp) / ap + 8.076e-3 * zp * ap ** (-0.66666)) / (1.0 + 4.871 * ap ** (-0.33333))
    epsp = -0.1724 * ap ** (-0.33333) + 3.051e-3 * zp ** 2 * ap ** (-1.33333) + 0.4166666 * deltap ** 2
    rp = 1.18 * ap ** 0.33333 * (1.0 + epsp)
    dp = 0.66666 * ((ap - 2.0 * zp) / ap - deltap) * rp
    rpp = rp - ((ap - zp) / ap) * dp
    cpp = rpp * (1.0 - (bpp / rpp) ** 2)
    rho0pp = 3.0 * zp / (12.5664 * cpp ** 3 * (1.0 + pi ** 2 * tpp ** 2 / (19.36 * cpp ** 2)))
    return rho0pp / (1.0 + math.exp((rg - cpp) / (tpp / 4.4)))


def rhonp(rg: float, ap: float, zp: float) -> float:
    pi = 3.141597265
    tnp = 2.4
    bnp = 0.413 * tnp
    deltap = ((ap - 2.0 * zp) / ap + 8.076e-3 * zp * ap ** (-0.66666)) / (1.0 + 4.871 * ap ** (-0.33333))
    epsp = -0.1724 * ap ** (-0.33333) + 3.051e-3 * zp ** 2 * ap ** (-1.33333) + 0.4166666 * deltap ** 2
    rp = 1.18 * ap ** 0.33333 * (1.0 + epsp)
    dp = 0.66666 * ((ap - 2.0 * zp) / ap - deltap) * rp
    rnp = rp + (zp / ap) * dp
    cnp = rnp * (1.0 - (bnp / rnp) ** 2)
    rho0np = 3.0 * (ap - zp) / (12.5664 * cnp ** 3 * (1.0 + pi ** 2 * tnp ** 2 / (19.36 * cnp ** 2)))
    return rho0np / (1.0 + math.exp((rg - cnp) / (tnp / 4.4)))


def derivative(wf: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """First and second derivative (6-point)."""
    npnts = wf.size - 1
    wf1 = np.zeros_like(wf)
    wf2 = np.zeros_like(wf)

    ds = 120.0 * dx
    A = np.array(
        [
            [-274.0 / ds, 600.0 / ds, -600.0 / ds, 400.0 / ds, -150.0 / ds, 24.0 / ds],
            [-24.0 / ds, -130.0 / ds, 240.0 / ds, -120.0 / ds, 40.0 / ds, -6.0 / ds],
            [6.0 / ds, -60.0 / ds, -40.0 / ds, 120.0 / ds, -30.0 / ds, 4.0 / ds],
            [-4.0 / ds, 30.0 / ds, -120.0 / ds, 40.0 / ds, 60.0 / ds, -6.0 / ds],
            [6.0 / ds, -40.0 / ds, 120.0 / ds, -240.0 / ds, 130.0 / ds, 24.0 / ds],
            [-24.0 / ds, 150.0 / ds, -400.0 / ds, 600.0 / ds, -600.0 / ds, 274.0 / ds],
        ]
    )

    ds2 = 60.0 * dx ** 2
    B = np.array(
        [
            [225.0 / ds2, -770.0 / ds2, 1070.0 / ds2, -780.0 / ds2, 305.0 / ds2, -50.0 / ds2],
            [50.0 / ds2, -75.0 / ds2, -20.0 / ds2, 70.0 / ds2, -30.0 / ds2, 5.0 / ds2],
            [-5.0 / ds2, 80.0 / ds2, -150.0 / ds2, 80.0 / ds2, -5.0 / ds2, 0.0 / ds2],
            [0.0 / ds2, -5.0 / ds2, 80.0 / ds2, -150.0 / ds2, 80.0 / ds2, -5.0 / ds2],
            [5.0 / ds2, -30.0 / ds2, 70.0 / ds2, -20.0 / ds2, -75.0 / ds2, 50.0 / ds2],
            [-50.0 / ds2, 305.0 / ds2, -780.0 / ds2, 1070.0 / ds2, -770.0 / ds2, 225.0 / ds2],
        ]
    )

    for ir in range(2, npnts - 2):
        wf1[ir] = (
            A[2, 0] * wf[ir - 2]
            + A[2, 1] * wf[ir - 1]
            + A[2, 2] * wf[ir]
            + A[2, 3] * wf[ir + 1]
            + A[2, 4] * wf[ir + 2]
            + A[2, 5] * wf[ir + 3]
        )
        wf2[ir] = (
            B[2, 0] * wf[ir - 2]
            + B[2, 1] * wf[ir - 1]
            + B[2, 2] * wf[ir]
            + B[2, 3] * wf[ir + 1]
            + B[2, 4] * wf[ir + 2]
            + B[2, 5] * wf[ir + 3]
        )

    # boundaries
    ir = 0
    wf1[ir] = A[0, 0] * wf[ir] + A[0, 1] * wf[ir + 1] + A[0, 2] * wf[ir + 2] + A[0, 3] * wf[ir + 3] + A[0, 4] * wf[ir + 4] + A[0, 5] * wf[ir + 5]
    wf2[ir] = B[0, 0] * wf[ir] + B[0, 1] * wf[ir + 1] + B[0, 2] * wf[ir + 2] + B[0, 3] * wf[ir + 3] + B[0, 4] * wf[ir + 4] + B[0, 5] * wf[ir + 5]

    ir = 1
    wf1[ir] = A[1, 0] * wf[ir - 1] + A[1, 1] * wf[ir] + A[1, 2] * wf[ir + 1] + A[1, 3] * wf[ir + 2] + A[1, 4] * wf[ir + 3] + A[1, 5] * wf[ir + 4]
    wf2[ir] = B[1, 0] * wf[ir - 1] + B[1, 1] * wf[ir] + B[1, 2] * wf[ir + 1] + B[1, 3] * wf[ir + 2] + B[1, 4] * wf[ir + 3] + B[1, 5] * wf[ir + 4]

    ir = npnts - 2
    wf1[ir] = A[3, 0] * wf[ir - 3] + A[3, 1] * wf[ir - 2] + A[3, 2] * wf[ir - 1] + A[3, 3] * wf[ir] + A[3, 4] * wf[ir + 1] + A[3, 5] * wf[ir + 2]
    wf2[ir] = B[3, 0] * wf[ir - 3] + B[3, 1] * wf[ir - 2] + B[3, 2] * wf[ir - 1] + B[3, 3] * wf[ir] + B[3, 4] * wf[ir + 1] + B[3, 5] * wf[ir + 2]

    ir = npnts - 1
    wf1[ir] = A[4, 0] * wf[ir - 4] + A[4, 1] * wf[ir - 3] + A[4, 2] * wf[ir - 2] + A[4, 3] * wf[ir - 1] + A[4, 4] * wf[ir] + A[4, 5] * wf[ir + 1]
    wf2[ir] = B[4, 0] * wf[ir - 4] + B[4, 1] * wf[ir - 3] + B[4, 2] * wf[ir - 2] + B[4, 3] * wf[ir - 1] + B[4, 4] * wf[ir] + B[4, 5] * wf[ir + 1]

    ir = npnts
    wf1[ir] = A[5, 0] * wf[ir - 5] + A[5, 1] * wf[ir - 4] + A[5, 2] * wf[ir - 3] + A[5, 3] * wf[ir - 2] + A[5, 4] * wf[ir - 1] + A[5, 5] * wf[ir]
    wf2[ir] = B[5, 0] * wf[ir - 5] + B[5, 1] * wf[ir - 4] + B[5, 2] * wf[ir - 3] + B[5, 3] * wf[ir - 2] + B[5, 4] * wf[ir - 1] + B[5, 5] * wf[ir]

    return wf1, wf2


# -----------------------------
# Numerical integration
# -----------------------------

def _integrate_trajectory(
    y0: np.ndarray,
    x1: float,
    x2: float,
    accuracy: float,
    initial_step: float,
    derivative: Callable[[float, np.ndarray, Context], np.ndarray],
    ctx: Context,
) -> np.ndarray:
    """Integrate one coupled-channel trajectory with SciPy's RK45 solver."""
    solution = solve_ivp(
        lambda x, y: derivative(x, y, ctx),
        (x1, x2),
        y0,
        method="RK45",
        # A conservative factor preserves the meaning of the documented ACCUR input
        # across the different embedded error estimators used by the two solvers.
        rtol=accuracy * 0.1,
        atol=accuracy * 1.0e-10,
        first_step=min(abs(initial_step), abs(x2 - x1)),
    )
    if not solution.success:
        raise RuntimeError(f"coupled-channel integration failed: {solution.message}")
    return solution.y[:, -1]


def _integrate_tabulated_curve(
    x: np.ndarray,
    y: np.ndarray,
    lower: float,
    upper: float,
    accuracy: float = 1.0e-6,
) -> float:
    """Integrate a natural cubic spline through tabulated real-valued data."""
    curve = CubicSpline(x, y, bc_type="natural")
    breakpoints = x[(x > lower) & (x < upper)]
    value, _ = quad(
        lambda point: float(curve(point)),
        lower,
        upper,
        epsabs=1.0e-10,
        epsrel=accuracy,
        points=breakpoints,
        limit=max(50, breakpoints.size + 1),
    )
    return value


# -----------------------------
# Core dynamical functions
# -----------------------------

def ylm(l: int, mm: int, theta: float, phi: float, fak: np.ndarray) -> complex:
    """Compute the normalized spherical harmonic Y_l^m(theta, phi)."""
    m = abs(mm)
    pi = 3.141597265
    x = math.cos(theta)
    if m > l or abs(x) > 1.0:
        raise RuntimeError("bad arguments in Ylm")
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for _ in range(1, m + 1):
            pmm = -pmm * fact * somx2
            fact = fact + 2.0
    if l == m:
        plgndr = pmm
    else:
        pmmp1 = x * (2 * m + 1) * pmm
        if l == m + 1:
            plgndr = pmmp1
        else:
            pll = 0.0
            for ll in range(m + 2, l + 1):
                pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
                pmm = pmmp1
                pmmp1 = pll
            plgndr = pll
    cephi = math.cos(phi) + 1j * math.sin(phi)
    y = math.sqrt((2.0 * l + 1.0) / (4.0 * pi) * fak[l - m] / fak[l + m]) * plgndr * cephi ** m
    if mm < 0:
        y = ((-1) ** m) * np.conj(y)
    return y


def vint(t: float, ctx: Context) -> np.ndarray:
    t2 = t * t
    phi = 1.0 / math.sqrt(1.0 + t2)
    phi3 = phi ** 3
    phi5 = phi ** 5
    beta2 = ctx.beta * ctx.beta
    gamma2 = ctx.gamma * ctx.gamma

    n = ctx.n
    vmat = np.zeros((n, n), dtype=np.complex128)

    for i in range(n - 1):
        for j in range(i + 1, n):
            mu = int(round(ctx.mm[j] - ctx.mm[i]))
            x = ctx.xi[j, i]

            # Electric dipole
            if mu == 0:
                q = math.sqrt(2.0) * ctx.gamma * (t * phi3 - 1j * x * beta2 * phi)
            else:
                q = -mu * phi3
            ve1 = ctx.psitot[0, i, j] * q

            # Electric quadrupole
            if abs(mu) == 2:
                q = 3.0 * phi5
            elif abs(mu) == 1:
                q = mu * ctx.gamma * (6.0 * t * phi5 - 1j * x * beta2 * phi3)
            else:
                q = math.sqrt(6.0) * gamma2 * ((2.0 * t2 - 1.0) * phi5 - 1j * beta2 * t * phi3)
            ve2 = ctx.psitot[1, i, j] * q

            # Magnetic dipole
            q = abs(mu) * 1j * ctx.beta * phi3
            vm1 = ctx.psitot[2, i, j] * q

            vmat[i, j] = ve1 + ve2 + vm1

            if ctx.iopnuc == 1:
                rr = ctx.bb / phi
                ir = int(rr / ctx.delr)
                h = 1.0 - (rr - ctx.delr * ir) / ctx.delr
                theta = t * phi
                if 0 < ir + 1 < ctx.ngrid:
                    vnuc = 0.0
                    for ln in range(0, 3):
                        if abs(mu) <= ln:
                            yval = ylm(ln, mu, theta, 0.0, ctx.fak)
                            vnuc = vnuc + (h * ctx.psinuc[ln, ir, i, j] + (1.0 - h) * ctx.psinuc[ln, ir + 1, i, j]) * yval
                    vmat[i, j] = vmat[i, j] + vnuc
            vmat[j, i] = np.conj(vmat[i, j])
        vmat[i, i] = 0.0
    vmat[n - 1, n - 1] = 0.0
    return vmat


def dcadt(tau: float, ca: np.ndarray, ctx: Context) -> np.ndarray:
    vmat = vint(tau, ctx)
    expt = np.exp(1j * tau * ctx.xi.T)
    dca = -1j * np.sum(vmat * expt * ca[None, :], axis=1)
    return dca


def phnuc(wn: np.ndarray, b: np.ndarray, z: np.ndarray, intr: np.ndarray, beta: float, hc: float, delr: float) -> np.ndarray:
    nb = b.size
    ngrid = z.size - 1
    phasen = np.zeros(nb, dtype=np.complex128)
    for k in range(nb):
        acc = 0.0 + 0.0j
        for i in range(ngrid + 1):
            rv = math.sqrt(b[k] ** 2 + z[i] ** 2)
            l = int(rv / delr)
            al = 1.0 - (rv - delr * l) / delr
            vnucl = 0.0 + 0.0j
            if 0 < l + 1 < ngrid:
                vnucl = al * wn[l] + (1.0 - al) * wn[l + 1]
            acc = acc + intr[i] * vnucl
        phasen[k] = -2.0 / hc / beta * delr / 3.0 * acc
    return phasen


def twofold(fact: complex, r: np.ndarray, delr: float, dens1: np.ndarray, dens2: np.ndarray, intr: np.ndarray) -> np.ndarray:
    ngrid = r.size - 1
    v = np.zeros(ngrid + 1, dtype=np.complex128)
    pi = 3.141597265
    delx = 2.0 / ngrid
    x = np.linspace(-1.0, 1.0, ngrid + 1)
    rint = r.copy()

    for i in range(ngrid + 1):
        sum2 = 0.0
        for j in range(ngrid + 1):
            sum1 = 0.0
            for k in range(ngrid + 1):
                aux = math.sqrt(abs(r[i] ** 2 + rint[j] ** 2 - 2.0 * r[i] * rint[j] * x[k]))
                ifrac = int(aux / delr)
                al = 1.0 - (aux - delr * ifrac) / delr
                if 0 < ifrac + 1 < ngrid:
                    sum1 = sum1 + intr[k] * (al * dens1[ifrac] + (1.0 - al) * dens1[ifrac + 1])
            ifrac = int(rint[j] / delr + 1.0)
            al = 1.0 - (rint[j] - delr * ifrac) / delr
            if 0 < ifrac + 1 < ngrid:
                sum2 = sum2 + intr[j] * rint[j] ** 2 * (al * dens2[ifrac] + (1.0 - al) * dens2[ifrac + 1]) * delx / 3.0 * sum1
        v[i] = 2.0 * pi * fact * sum2 / 3.0 * delr
    return v


# -----------------------------
# Main driver
# -----------------------------

def run_relex(inputs: dict) -> RelexResult:
    # Input extraction
    system = inputs["system"]
    integ = inputs["integration"]
    options = inputs["options"]
    grid = inputs.get("grid", {})

    ap = float(system["ap"])
    zp = float(system["zp"])
    at = float(system["at"])
    zt = float(system["zt"])
    eca = float(system["eca"])
    iw = int(system.get("iw", 0))
    iout = int(system.get("iout", 0))

    nb = int(integ["nb"])
    accur = float(integ["accur"])
    bmin = float(integ["bmin"])
    itot = int(integ.get("itot", 0))

    iopw = int(options.get("iopw", 0))
    iopnuc = int(options.get("iopnuc", 0))

    ngrid = int(grid.get("ngrid", 200))
    if ngrid % 2 != 0:
        raise ValueError("ngrid must be even")

    # Store the collision partner generating the excitation field in (ap, zp).
    if iw == 0:
        ap, at = at, ap
        zp, zt = zt, zp

    # Constants
    ei = 1j
    pi = math.acos(-1.0)
    e2 = 1.44
    amu = 931.5
    hc = 197.33
    signn = 4.0

    # Factorials and signs
    iv, fak, fad = gfv(130)

    # Read states
    states = inputs["states"]
    nst = len(states)
    ex = np.zeros(nst, dtype=float)
    spin = np.zeros(nst, dtype=float)
    for s in states:
        idx = int(s["i"]) - 1
        ex[idx] = float(s["ex"])
        spin[idx] = float(s["spin"])

    # Matrix elements
    mat = np.zeros((3, nst, nst), dtype=float)
    for me in inputs.get("matrix_elements", []):
        i = int(me["i"]) - 1
        j = int(me["j"]) - 1
        mat[0, i, j] = float(me.get("mate1", 0.0))
        mat[1, i, j] = float(me.get("mate2", 0.0))
        mat[2, i, j] = float(me.get("matm1", 0.0))

    # Nuclear deformations
    delte = np.zeros((3, nst), dtype=float)
    if iopnuc == 1:
        for d in inputs.get("nuclear_deformation", []):
            j = int(d["j"]) - 1
            delte[0, j] = float(d.get("delte0", 0.0))
            delte[1, j] = float(d.get("delte1", 0.0))
            delte[2, j] = float(d.get("delte2", 0.0))

    # Construct magnetic substates
    mj = np.rint(2.0 * spin + 1.0).astype(int)
    ii = []
    jj = []
    mm = []
    exx = []
    istop = [0]
    for i in range(nst):
        for m in range(mj[i]):
            ii.append(i)
            jj.append(spin[i])
            mm.append(-spin[i] + float(m))
            exx.append(ex[i])
        istop.append(len(ii))
    ii = np.array(ii, dtype=int)
    jj = np.array(jj, dtype=float)
    mm = np.array(mm, dtype=float)
    exx = np.array(exx, dtype=float)
    n = len(ii)

    # Lorentz variables
    gamma = 1.0 + eca / amu
    beta = math.sqrt((gamma - 1.0) * (gamma + 1.0)) / gamma

    # Grazing impact parameter and mesh
    rp13 = 1.2 * ap ** (1.0 / 3.0)
    rt13 = 1.2 * at ** (1.0 / 3.0)
    b13 = rp13 + rt13
    bmin1 = b13 / 2.0
    bmax1 = 1.5 * b13
    nb1 = int(nb / 2)
    if iv[nb1] >= 0.0:
        nb1 = nb1 + 1
    delb1 = (bmax1 - bmin1) / (nb1 - 1)
    b = np.zeros(nb, dtype=float)
    for ib in range(nb1):
        b[ib] = bmin1 + ib * delb1
    nb2 = nb - nb1
    bmax = 200.0
    delb2 = (bmax - bmax1) / nb2
    for ib in range(nb1, nb):
        b[ib] = bmax1 + (ib - nb1 + 1) * delb2

    # r,z meshes
    rmax = 30.0
    delr = rmax / ngrid
    r = np.array([i * delr for i in range(ngrid + 1)], dtype=float)
    z = r.copy()

    # Simpson integration factors
    intr = np.zeros(ngrid + 1, dtype=int)
    intr[0] = 1
    intr[ngrid] = 1
    ig = 4
    for i in range(1, ngrid):
        intr[i] = ig
        ig = 6 - ig

    # Optical potential
    wn = np.zeros(ngrid + 1, dtype=np.complex128)
    if iopw == 1:
        opt = inputs.get("optical_potential", {})
        ru, u = _read_optical_potential(opt)
        dru = np.zeros_like(ru)
        for i in range(1, ru.size):
            dru[i] = ru[i] - ru[i - 1]
        if ru.size > 1:
            u[0] = u[1]
        # Linearly interpolate the supplied optical potential onto the radial mesh.
        ncount = 0
        for ir in range(1, ru.size - 1):
            for i in range(ncount, ngrid + 1):
                aux = ru[ir] - r[i]
                if aux >= 0.0:
                    al = 1.0 - aux / dru[ir]
                    wn[i] = al * u[ir] + (1.0 - al) * u[ir - 1]
                    ncount += 1
                else:
                    break
    else:
        # Compute liquid-drop densities
        dens1 = np.zeros(ngrid + 1, dtype=float)
        dens2 = np.zeros(ngrid + 1, dtype=float)
        for i in range(ngrid + 1):
            dens1[i] = rhonp(r[i], ap, zp) + rhopp(r[i], ap, zp)
            dens2[i] = rhonp(r[i], at, zt) + rhopp(r[i], at, zt)
        fact = -1j * signn * hc * beta / 2.0
        wn = twofold(fact, r, delr, dens1, dens2, intr)

    # Nuclear interaction (Bohr-Mottelson)
    psinuc = np.zeros((3, ngrid + 1, n, n), dtype=np.complex128)
    if iopnuc == 1:
        wn1, wn2 = derivative(wn, delr)
        ubm = np.zeros((3, ngrid + 1), dtype=np.complex128)
        ubm[0, :] = 3.0 * wn + r * wn1
        ubm[1, :] = 1.5 * (wn1 + r * wn2 / 3.0)
        ubm[2, :] = wn1
    else:
        ubm = np.zeros((3, ngrid + 1), dtype=np.complex128)

    # Nuclear eikonal phase
    phasen = phnuc(wn, b, z, intr, beta, hc, delr)
    tb = np.exp(-2.0 * np.imag(phasen))

    # psi matrix and nuclear excitation matrices
    fac = zp * e2 / beta / hc
    fd1 = fac * math.sqrt(2.0 * pi / 3.0)
    fd2 = -fac * math.sqrt(2.0 * pi / 15.0) / 2.0

    psi = np.zeros((3, nb, n, n), dtype=np.complex128)

    for i in range(n - 1):
        for j in range(i + 1, n):
            mu = int(round(mm[j] - mm[i]))
            for l_idx in range(3):
                ll = 1 if l_idx in (0, 2) else 2
                bmat = mat[l_idx, ii[i], ii[j]]
                iv1_idx = int(round(jj[j] + ll - mm[j] + 1.0))
                iv1 = iv[iv1_idx] if 0 <= iv1_idx < iv.size else 0.0
                three = threej(jj[j], float(ll), jj[i], -mm[j], float(mu), mm[i])
                aux = bmat * iv1 * three
                if ll == 1:
                    psi[l_idx, :, i, j] = fd1 * aux / (b ** ll)
                else:
                    psi[l_idx, :, i, j] = fd2 * aux / (b ** ll)
            if iopnuc == 1:
                for ln in range(3):
                    bmatn = delte[ln, ii[j]]
                    iv2_idx = int(round(jj[j] - mm[j]))
                    iv2 = iv[iv2_idx] if 0 <= iv2_idx < iv.size else 0.0
                    zero = 0.0
                    three = threej(jj[j], float(ln), jj[i], -mm[j], float(mu), mm[i])
                    three0 = threej(jj[j], float(ln), jj[i], zero, zero, zero)
                    auxn = bmatn * math.sqrt((2.0 * jj[i] + 1.0) / (2.0 * jj[j] + 1.0) / 4.0 / pi) * iv2 * three * three0 * gamma
                    psinuc[ln, :, i, j] = psinuc[ln, :, i, j] + auxn * ubm[ln, :]

    # Output containers
    spinave = np.zeros((nst, nb), dtype=float)
    stat_by_b: List[np.ndarray] = []

    # Integration loop over impact parameter
    for ib in range(nb):
        e0 = hc * beta * gamma / b[ib]
        xi = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                xi[i, j] = (exx[i] - exx[j]) / e0
        psitot = np.zeros((3, n, n), dtype=np.complex128)
        for l_idx in range(3):
            psitot[l_idx, :, :] = psi[l_idx, ib, :, :]

        # Initialize statistical tensors
        stat = np.zeros((nst, 5, 9), dtype=np.complex128)

        # Average over initial orientation
        # One-based magnetic-substate boundaries for the tensor summation.
        ii_f = np.zeros(n + 1, dtype=int)
        ii_f[0] = 0
        ii_f[1:] = ii + 1
        istop_f = np.array(istop, dtype=int)

        for mj1 in range(mj[0]):
            ca = np.zeros(n, dtype=np.complex128)
            ca[mj1] = 1.0 + 0.0j

            # integrate coupled-channels equations
            tmax = 15.0
            nt = 50
            dtau = 2.0 * tmax / nt
            t1 = -tmax
            t2 = tmax
            h1 = dtau
            ctx = Context(
                beta=beta,
                gamma=gamma,
                ei=ei,
                pi=pi,
                hc=hc,
                xi=xi,
                mm=mm,
                n=n,
                psitot=psitot,
                psinuc=psinuc,
                bb=b[ib],
                iopnuc=iopnuc,
                delr=delr,
                ngrid=ngrid,
                iv=iv,
                fak=fak,
            )

            ca = _integrate_trajectory(ca, t1, t2, accur, h1, dcadt, ctx)

            # excitation probabilities
            pcc_vals = np.zeros(nst, dtype=float)
            for j in range(nst):
                start = istop[j]
                end = istop[j + 1]
                sum_p = 0.0
                for i in range(start, end):
                    sum_p += abs(ca[i]) ** 2
                pcc = sum_p * tb[ib]
                pcc_vals[j] = pcc
                spinave[j, ib] += pcc / mj[0]

            # norm check
            xn = float(np.sum(pcc_vals)) + 1.0 - tb[ib]
            if abs(xn - 1.0) > accur * 10.0:
                raise RuntimeError("Error in probability exceeds 10 x ACCUR")

            # statistical tensors
            if iout == 1:
                for ka in (0, 2, 4):
                    for kappa in range(-ka, ka + 1):
                        for i_state in range(1, nst + 1):
                            # Determine this state's one-based magnetic-substate bounds.
                            jini_f = istop_f[ii_f[i_state - 1]] + 1
                            jfin_f = istop_f[ii_f[i_state]]
                            for j_f in range(jini_f, jfin_f + 1):
                                j = j_f - 1
                                mnkap = mm[j] + kappa
                                iv3_idx = int(round(abs(jj[j] + mnkap)))
                                iv3 = iv[iv3_idx] if 0 <= iv3_idx < iv.size else 0.0
                                three = threej(jj[j], jj[j], float(ka), -mnkap, mm[j], float(kappa))
                                jkap_f = j_f + kappa
                                if jini_f <= jkap_f <= jfin_f:
                                    jkap = jkap_f - 1
                                    stat[i_state - 1, ka, kappa + 4] += iv3 * three * math.sqrt(2.0 * jj[j] + 1.0) * np.conj(ca[j]) * ca[jkap] / mj[0]

        if iout == 1:
            stat_by_b.append(stat)

    # Cross sections
    cross_sections: Dict[int, float] = {}
    ared = ap * at / (ap + at)
    aclose = zp * zt * e2 / (ared * amu * beta * beta)
    bc = pi * aclose / 2.0 / gamma
    for k in range(1, nst):
        bcorr = b - bc
        pint = 20.0 * pi * b * spinave[k, :]
        cross = _integrate_tabulated_curve(bcorr, pint, max(bmin1, bmin), bmax)
        cross_sections[k + 1] = cross

    return RelexResult(inputs=inputs, b=b, spinave=spinave, cross_sections=cross_sections, stat_by_b=stat_by_b)


def _read_optical_potential(opt: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Load optical potential from file path or inline arrays."""
    if "path" in opt:
        path = opt["path"]
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise ValueError("optical potential file is empty")
        nr = int(lines[0].split()[0])
        ru = np.zeros(nr + 1, dtype=float)
        u = np.zeros(nr + 1, dtype=np.complex128)
        for i in range(1, min(nr, len(lines) - 1) + 1):
            parts = lines[i].split()
            if len(parts) < 3:
                raise ValueError("optical potential line must have r, real, imag")
            ru[i] = float(parts[0])
            u[i] = float(parts[1]) + 1j * float(parts[2])
        return ru, u
    if "r" in opt and "real" in opt and "imag" in opt:
        r = opt["r"]
        real = opt["real"]
        imag = opt["imag"]
        if not (len(r) == len(real) == len(imag)):
            raise ValueError("optical potential arrays must have equal length")
        nr = len(r)
        ru = np.zeros(nr + 1, dtype=float)
        u = np.zeros(nr + 1, dtype=np.complex128)
        for i in range(1, nr + 1):
            ru[i] = float(r[i - 1])
            u[i] = float(real[i - 1]) + 1j * float(imag[i - 1])
        return ru, u
    raise ValueError("optical_potential must provide 'path' or arrays ('r','real','imag')")
