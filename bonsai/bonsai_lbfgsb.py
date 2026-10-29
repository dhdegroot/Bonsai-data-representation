# Adapted from scipy 1.17.1 scipy/optimize/_lbfgsb_py.py, _differentiable_functions.py and _optimize.py.
#
# scipy is distributed under the following notice:
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# scipy/optimize/_lbfgsb_py.py also carries the following notice:
#
# Copyright (c) 2004 David M. Cooke <cookedm@physics.mcmaster.ca>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Modifications by Travis Oliphant and Enthought, Inc. for inclusion in SciPy

"""
L-BFGS-B without scipy.optimize.minimize's per-call set-up.

Bonsai solves tens of thousands of small bounded problems per reconstruction, and for these minimize spends longer
building its ScalarFunction, MemoizeJac, bounds and result objects than the objective takes. minimizeLBFGSB makes the
same calls to scipy's L-BFGS-B routine setulb that minimize makes, with the same arguments, and evaluates the
objective at the same points, so its result is identical bit for bit. It is written against scipy 1.17.1, the
version requirements.txt pins; on any other version it returns None and the caller uses minimize.
"""
import math

import numpy as np
import scipy
from scipy.optimize import OptimizeResult

if scipy.__version__ == '1.17.1':
    from scipy.optimize._lbfgsb import setulb as _setulb
else:
    _setulb = None

# scipy's defaults for minimize(method='L-BFGS-B'); ftol and gtol apply when tol is None
_MAXCOR = 10
_MAXLS = 20
_MAXFUN = 15000
_FTOL = 2.2204460492503131e-09
_GTOL = 1e-5

# setulb's task codes (task[0], and task[1] for the stops set here)
_TASK_NEW_X = 1
_TASK_FG = 3
_TASK_CONVERGENCE = 4
_TASK_STOP = 5
_STOP_MAXFUN = 502
_STOP_MAXITER = 504


class _Objective:
    """
    Evaluates fun(x, *args), which returns the value and the gradient, at the points and as many times as scipy
    1.17.1's ScalarFunction and MemoizeJac evaluate it for minimize(jac=True).
    """

    def __init__(self, fun, args):
        self.fun = fun
        self.args = args
        self.nfev = 0
        self.lowestValue = np.inf
        self.point = None
        self.value = None
        self.grad = None

    def valueAndGrad(self, x):
        # A point is new unless every element equals the last point's (0.0 equals -0.0, NaN equals nothing), as
        # ScalarFunction decides with np.array_equal.
        point = x.tolist()
        if point != self.point:
            self.point = point
            self._evaluate(x, point)
        return self.value, self.grad

    def _evaluate(self, x, point):
        valueAndGrad = self.fun(x.copy(), *self.args)
        grad = valueAndGrad[1]
        value = valueAndGrad[0]
        if not np.isscalar(value):
            try:
                value = np.asarray(value).item()
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function must return a scalar value.") from e
        self.nfev += 1
        # ScalarFunction keeps the lowest value seen; the comparison raises TypeError for None or complex values.
        if value < self.lowestValue:
            self.lowestValue = value
        # MemoizeJac calls fun again, uncounted, when the gradient was None or x holds a NaN (x == x fails).
        if grad is None or any(map(math.isnan, point)):
            grad = self.fun(x.copy(), *self.args)[1]
        self.value = value
        self.grad = np.atleast_1d(grad)


def minimizeLBFGSB(fun, x0, args, bounds, tol, maxiter=15000):
    """
    Returns what scipy.optimize.minimize(fun, x0, args=args, jac=True, bounds=bounds, tol=tol,
    options={'maxiter': maxiter}) returns with method L-BFGS-B, bit for bit, as an OptimizeResult holding fun, x,
    success and nit.

    :param fun: returns the objective value and its gradient at x
    :param args: tuple of extra arguments to fun
    :param bounds: a (lower, upper) pair of numbers for every variable
    :return: the result, or None when scipy is not version 1.17.1 or minimize would take another path (a bound that
    is not finite, a lower bound above its upper bound, every variable fixed, x0 not a float64 vector as long as
    bounds). The caller then calls minimize itself; fun has not been called.
    """
    if _setulb is None:
        return None
    x0 = np.atleast_1d(np.asarray(x0))
    n = len(bounds)
    if x0.shape != (n,) or x0.dtype != np.float64:
        return None
    lb, ub = np.array(bounds, dtype=np.float64).reshape(n, 2).T.copy()
    if not (np.isfinite(lb) & np.isfinite(ub)).all() or (lb > ub).any() or (lb == ub).all():
        return None

    if tol is None:
        ftol, gtol = _FTOL, _GTOL
    else:
        ftol = gtol = tol
    factr = ftol / np.finfo(float).eps
    x = np.clip(x0, lb, ub)
    objective = _Objective(fun, args)
    # minimize evaluates fun at the clipped x0 before setulb first asks for it
    objective.valueAndGrad(x)

    f = np.array(0.0)
    g = np.zeros(n)
    nbd = np.full(n, 2, dtype=np.int32)
    wa = np.zeros(2 * _MAXCOR * n + 5 * n + 11 * _MAXCOR * _MAXCOR + 8 * _MAXCOR)
    iwa = np.zeros(3 * n, dtype=np.int32)
    task = np.zeros(2, dtype=np.int32)
    lnTask = np.zeros(2, dtype=np.int32)
    lsave = np.zeros(4, dtype=np.int32)
    isave = np.zeros(44, dtype=np.int32)
    dsave = np.zeros(29)
    nit = 0
    while True:
        _setulb(_MAXCOR, x, lb, ub, nbd, f, g, factr, gtol, wa, iwa, task, lsave, isave, dsave, _MAXLS, lnTask)
        if task[0] == _TASK_FG:
            f, g = objective.valueAndGrad(x)
            # setulb writes into g (after a failed line search), and a repeated point must return fun's gradient
            g = g.astype(np.float64)
        elif task[0] == _TASK_NEW_X:
            nit += 1
            if nit >= maxiter:
                task[:] = _TASK_STOP, _STOP_MAXITER
            elif objective.nfev > _MAXFUN:
                task[:] = _TASK_STOP, _STOP_MAXFUN
        else:
            return OptimizeResult(fun=f, x=x, success=bool(task[0] == _TASK_CONVERGENCE), nit=nit)
