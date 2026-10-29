"""Compiled kernels behind Bonsai's likelihood and diffusion-time computations.

Each kernel repeats, bit for bit, the numpy code it stands in for: the same operations in the same order, numpy's
pairwise summation wherever that code calls np.sum, and no fastmath, so nothing is reassociated or fused into a
multiply-add. Logarithms stay with numpy, whose SIMD log differs from libm's in the last bit: kernels fill buffers
that the caller passes to np.log.

A kernel returns FALLBACK wherever the numpy code would warn (a value that is not finite, the log of a value that is
not positive); the caller then runs the numpy code, which warns as before. Kernels that can see an underflow add TINY
to their status when a product or quotient is below 2**-1021 in magnitude, where numpy reports an underflow if
np.seterr asks for it.
"""

import math

import numba
import numpy as np
from numba import types

DONE = 0
FALLBACK = 1
AT_ZERO = 2
GAVE_UP = 3
AS_NODE = 4
TINY = 8

_TINY_MAGNITUDE = 2.0 ** -1021

# scipy.optimize.brentq's defaults, which getOptTime2LeafTree relies on, and the smallest rtol it accepts
_BRENTQ_XTOL = 2e-12
_BRENTQ_MAXITER = 100
_BRENTQ_MIN_RTOL = 4 * np.finfo(np.float64).eps

_float = types.float64
_int = types.int64
_vector = types.Array(types.float64, 1, 'C', readonly=True)
_stridedVector = types.Array(types.float64, 1, 'A', readonly=True)
_matrix = types.Array(types.float64, 2, 'C', readonly=True)
_indices = types.Array(types.int64, 1, 'C', readonly=True)
_outVector = types.float64[::1]
_outMatrix = types.float64[:, ::1]


# error_model='numpy' gives inf and NaN where numpy does instead of raising ZeroDivisionError. cache=True keeps the
# compiled code beside this file (or in NUMBA_CACHE_DIR), so only the first import after an install compiles; the
# explicit signatures compile at that import.
def _kernel(signature):
    return numba.njit(signature, cache=True, error_model='numpy')


_helper = numba.njit(cache=True, error_model='numpy')


@_helper
def _notFinite(x):
    return x - x != 0.


@_helper
def _notPositive(x):
    # True for NaN and infinity as well
    return (x <= 0.) | (x - x != 0.)


@_helper
def _isTiny(x):
    return abs(x) < _TINY_MAGNITUDE


@_helper
def _tinyFlag(tiny):
    return TINY if tiny else 0


# numpy's summation


@_helper
def _newSumStack():
    """Workspace for _pairwiseSum: pending (start, length) ranges and partial sums."""
    return np.empty((128, 2), np.int64), np.empty(64)


# Inlined into its callers: _rowSums calls it once per row
@numba.njit(cache=True, error_model='numpy', inline='always')
def _blockSum(a, start, n):
    """pairwise_sum of a[start:start + n] for n <= 128: one block, without recursion."""
    if n < 8:
        total = 0.
        for i in range(start, start + n):
            total += a[i]
        return total
    r0 = a[start]
    r1 = a[start + 1]
    r2 = a[start + 2]
    r3 = a[start + 3]
    r4 = a[start + 4]
    r5 = a[start + 5]
    r6 = a[start + 6]
    r7 = a[start + 7]
    i = 8
    while i < n - n % 8:
        r0 += a[start + i]
        r1 += a[start + i + 1]
        r2 += a[start + i + 2]
        r3 += a[start + i + 3]
        r4 += a[start + i + 4]
        r5 += a[start + i + 5]
        r6 += a[start + i + 6]
        r7 += a[start + i + 7]
        i += 8
    total = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        total += a[start + i]
        i += 1
    return total


@_helper
def _pairwiseSum(a, n, stack):
    """numpy's pairwise_sum (numpy/_core/src/umath/loops_utils.h.src) of a[:n]. Its recursion runs on an explicit
    stack: numba aborts a process that loads a self-recursive function from its cache."""
    if n <= 128:
        return _blockSum(a, 0, n)
    ranges, partialSums = stack
    ranges[0, 0] = 0
    ranges[0, 1] = n
    nRanges = 1
    nSums = 0
    while nRanges > 0:
        nRanges -= 1
        rangeStart = ranges[nRanges, 0]
        rangeLength = ranges[nRanges, 1]
        if rangeLength < 0:
            # Both halves of this range are summed: add them, left half first
            nSums -= 1
            partialSums[nSums - 1] = partialSums[nSums - 1] + partialSums[nSums]
        elif rangeLength <= 128:
            partialSums[nSums] = _blockSum(a, rangeStart, rangeLength)
            nSums += 1
        else:
            half = rangeLength // 2
            half -= half % 8
            ranges[nRanges, 1] = -1
            ranges[nRanges + 1, 0] = rangeStart + half
            ranges[nRanges + 1, 1] = rangeLength - half
            ranges[nRanges + 2, 0] = rangeStart
            ranges[nRanges + 2, 1] = half
            nRanges += 3
    return partialSums[0]


@_helper
def _npSum(a, n, stack):
    """np.sum of a[:n] for a contiguous float64 vector a."""
    return 0. + _pairwiseSum(a, n, stack)


@_helper
def _rowSums(m_gk, nColumns, sums_g, stack):
    """np.sum(m_gk[:, :nColumns], axis=1) of a C-ordered matrix into sums_g. Rows of up to 128 are summed without a
    call: a call per row costs more than the sum."""
    if nColumns < 8:
        for g in range(m_gk.shape[0]):
            total = 0.
            for c in range(nColumns):
                total += m_gk[g, c]
            sums_g[g] = 0. + total
    elif nColumns <= 128:
        for g in range(m_gk.shape[0]):
            sums_g[g] = 0. + _blockSum(m_gk[g], 0, nColumns)
    else:
        for g in range(m_gk.shape[0]):
            sums_g[g] = 0. + _pairwiseSum(m_gk[g], nColumns, stack)


# Two-leaf tree: getOptTime2LeafTree, der2LeafTree and getLoglik2LeafTree


@_helper
def _der2LeafTree(t12, summedLtqsVars_g, sqDists_g, terms_g, stack):
    """der2LeafTree's np.sum((-1 + sqDists_g / totVar_g) / totVar_g) with totVar_g = t12 + summedLtqsVars_g; returns
    the derivative, whether a value is not finite and whether one is tiny."""
    notFinite = False
    tiny = False
    for g in range(terms_g.shape[0]):
        totVar = t12 + summedLtqsVars_g[g]
        quotient = sqDists_g[g] / totVar
        term = (-1. + quotient) / totVar
        terms_g[g] = term
        notFinite |= _notFinite(totVar)
        tiny |= _isTiny(quotient) | _isTiny(term)
    der = _npSum(terms_g, terms_g.shape[0], stack)
    return der, notFinite | _notFinite(der), tiny


@_helper
def _signbit(x):
    return math.copysign(1., x) < 0.


# Adapted from scipy 1.17.1 scipy/optimize/Zeros/brentq.c, under the following notice:
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
@_helper
def _brentq(xa, xb, fa, fb, rtol, summedLtqsVars_g, sqDists_g, terms_g, stack):
    """scipy 1.17.1's brentq (scipy/optimize/Zeros/brentq.c) on der2LeafTree, line for line, given the derivative at
    both ends of the bracket; returns (status, root, tiny)."""
    tiny = False
    xpre = xa
    xcur = xb
    xblk = 0.
    fpre = fa
    fcur = fb
    fblk = 0.
    spre = 0.
    scur = 0.
    if fpre == 0:
        return DONE, xpre, tiny
    if fcur == 0:
        return DONE, xcur, tiny
    if _signbit(fpre) == _signbit(fcur):
        return FALLBACK, 0., tiny
    for i in range(_BRENTQ_MAXITER):
        if fpre != 0 and fcur != 0 and _signbit(fpre) != _signbit(fcur):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (_BRENTQ_XTOL + rtol * abs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or abs(sbis) < delta:
            return DONE, xcur, tiny

        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                # interpolate
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            minStep = abs(spre) if abs(spre) < 3 * abs(sbis) - delta else 3 * abs(sbis) - delta
            if 2 * abs(stry) < minStep:
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += (delta if sbis > 0 else -delta)

        fcur, notFinite, tinyHere = _der2LeafTree(xcur, summedLtqsVars_g, sqDists_g, terms_g, stack)
        if notFinite:
            return FALLBACK, 0., tiny
        tiny |= tinyHere
    # brentq raises when it does not converge
    return FALLBACK, 0., tiny


@_helper
def _optTime2LeafTree(ltqs1_g, ltqsVars1_g, ltqs2_g, ltqsVars2_g, tol, summedLtqsVars_g, sqDists_g, terms_g):
    """getOptTime2LeafTree: the bracket search from [0, 1], then brentq. Returns (status, time), the status AT_ZERO
    for its (0, True), GAVE_UP for (None, False) and DONE for a converged root; summedLtqsVars_g and sqDists_g are
    left holding ltqsVars1_g + ltqsVars2_g and (ltqs1_g - ltqs2_g) ** 2."""
    nGenes = ltqs1_g.shape[0]
    if ltqsVars1_g.shape[0] != nGenes or ltqs2_g.shape[0] != nGenes or ltqsVars2_g.shape[0] != nGenes:
        return FALLBACK, 0.
    tiny = False
    for g in range(nGenes):
        summedLtqsVars_g[g] = ltqsVars1_g[g] + ltqsVars2_g[g]
        diff = ltqs1_g[g] - ltqs2_g[g]
        sqDists_g[g] = diff * diff
        tiny |= _isTiny(sqDists_g[g])
    stack = _newSumStack()

    lbBracket = 0.
    ubBracket = 1.
    valueLb, notFinite, tinyHere = _der2LeafTree(lbBracket, summedLtqsVars_g, sqDists_g, terms_g, stack)
    tiny |= tinyHere
    if notFinite:
        return FALLBACK, 0.
    if valueLb <= 0:
        return AT_ZERO | _tinyFlag(tiny), 0.
    valueUb, notFinite, tinyHere = _der2LeafTree(ubBracket, summedLtqsVars_g, sqDists_g, terms_g, stack)
    tiny |= tinyHere
    if notFinite:
        return FALLBACK, 0.
    bracketCounter = 0
    while valueUb >= 0:
        bracketCounter += 1
        if bracketCounter > 10:
            # The Python version prints a message here
            return FALLBACK, 0.
        lbBracket = ubBracket
        valueLb = valueUb
        ubBracket *= 10
        valueUb, notFinite, tinyHere = _der2LeafTree(ubBracket, summedLtqsVars_g, sqDists_g, terms_g, stack)
        tiny |= tinyHere
        if notFinite:
            return FALLBACK, 0.
        if lbBracket > 1e6:
            return GAVE_UP | _tinyFlag(tiny), 0.
        if valueUb > 1e9:
            # The Python version exits here
            return FALLBACK, 0.
    if not tol >= _BRENTQ_MIN_RTOL:
        return FALLBACK, 0.
    # brentq evaluates the derivative at both ends again; the bracket search already has those values
    status, root, tinyHere = _brentq(lbBracket, ubBracket, valueLb, valueUb, tol, summedLtqsVars_g, sqDists_g,
                                     terms_g, stack)
    return status | _tinyFlag(tiny | tinyHere), root


@_kernel(types.Tuple((_int, _float))(_stridedVector, _stridedVector, _stridedVector, _stridedVector, _float,
                                     types.boolean, _outVector, _outVector))
def optTime2LeafTree(ltqs1_g, ltqsVars1_g, ltqs2_g, ltqsVars2_g, tol, withLoglikTerms, totalVars_g, quotients_g):
    """getOptTime2LeafTree(ltqs1_g, ltqsVars1_g, ltqs2_g, ltqsVars2_g, tol): (status, time). With withLoglikTerms,
    also getLoglik2LeafTree's terms at the time TreeNode.get_dlogl_spr_move uses (the root; 0 at the lower bound; 1
    when the search gave up): totalVars_g = ltqsVars1_g + ltqsVars2_g + t, for np.log, and quotients_g =
    (ltqs1_g - ltqs2_g) ** 2 / totalVars_g. Without, both vectors are scratch."""
    nGenes = ltqs1_g.shape[0]
    if totalVars_g.shape[0] != nGenes or quotients_g.shape[0] != nGenes:
        return FALLBACK, 0.
    status, optTime = _optTime2LeafTree(ltqs1_g, ltqsVars1_g, ltqs2_g, ltqsVars2_g, tol, totalVars_g, quotients_g,
                                        np.empty(nGenes))
    if not withLoglikTerms or status == FALLBACK:
        return status, optTime
    if (status & ~TINY) == AT_ZERO:
        t = 0.
    elif (status & ~TINY) == GAVE_UP:
        t = 1.
    else:
        t = optTime
    invalid = False
    tiny = False
    for g in range(nGenes):
        totalVar = totalVars_g[g] + t
        quotient = quotients_g[g] / totalVar
        totalVars_g[g] = totalVar
        quotients_g[g] = quotient
        invalid |= _notPositive(totalVar) | _notFinite(quotient)
        tiny |= _isTiny(quotient)
    if invalid:
        return FALLBACK, 0.
    return status | _tinyFlag(tiny), optTime


@_kernel(_float(_outVector, _vector))
def loglik2LeafTree(logTotalVars_g, quotients_g):
    """getLoglik2LeafTree's -np.sum(np.log(totalVars_g) + sqDists_g / totalVars_g), given the logs and the quotients;
    logTotalVars_g is overwritten."""
    nGenes = logTotalVars_g.shape[0]
    if quotients_g.shape[0] != nGenes:
        return np.nan
    for g in range(nGenes):
        logTotalVars_g[g] = logTotalVars_g[g] + quotients_g[g]
    return -_npSum(logTotalVars_g, nGenes, _newSumStack())


# Compiled on first use: only the Python fallbacks and the recursive Tree.optTimes objective call it
@_helper
def der2LeafTree(t12, summedLtqsVars_g, sqDists_g):
    """der2LeafTree(t12, summedLtqsVars_g, sqDists_g): (status, derivative)."""
    nGenes = summedLtqsVars_g.shape[0]
    if sqDists_g.shape[0] != nGenes:
        return FALLBACK, 0.
    der, notFinite, tiny = _der2LeafTree(t12, summedLtqsVars_g, sqDists_g, np.empty(nGenes), _newSumStack())
    if notFinite:
        return FALLBACK, 0.
    return DONE | _tinyFlag(tiny), der


# Three-leaf star trees: the objectives of optimiseT3LeafStarSequential and optimiseT3LeafStar, and calcSingleDLogL


@_kernel(_int(_matrix, _matrix, _vector, types.boolean, _outMatrix, _outVector, _outVector))
def starTreeTerms(ltqs_gi, ltqsVars_gi, t_i, sequential, logBuffer_g4, sqDists_g, negGrad):
    """getLoglikAndGradStarTree(ltqs_gi=..., ltqsVars_gi=..., t_i=..., returnGrad=True) for three leafs, per gene in
    numpy's order: [w0 | w1 | w2 | W] into logBuffer_g4 for np.log, sum_i w_i (xr - l_i) ** 2 into sqDists_g, and
    the negated gradient into negGrad: getLogLikAndGradStarTreeSequentialWrapper's [-(g0 - g1), -(t2 g2)] when
    sequential, else getLogLikAndGradStarTreeWrapper's -t_i * g. It sets no TINY flag: callers use it only while
    np.geterr()['under'] is 'ignore'."""
    nGenes = ltqs_gi.shape[0]
    if (nGenes == 0 or ltqs_gi.shape[1] != 3 or ltqsVars_gi.shape[0] != nGenes or ltqsVars_gi.shape[1] != 3
            or t_i.shape[0] != 3 or logBuffer_g4.shape[0] != nGenes or logBuffer_g4.shape[1] != 4
            or sqDists_g.shape[0] != nGenes or negGrad.shape[0] != (2 if sequential else 3)):
        return FALLBACK
    t0 = t_i[0]
    t1 = t_i[1]
    t2 = t_i[2]
    # The gene loop vectorizes only over contiguous vectors, so the leafs' columns are copied out first; the gradient
    # sums run over genes in order, as numpy's axis-0 sum does, in a loop of their own
    columns = np.empty((10, nGenes))
    l0_g = columns[0]
    l1_g = columns[1]
    l2_g = columns[2]
    w0_g = columns[3]
    w1_g = columns[4]
    w2_g = columns[5]
    W_g = columns[6]
    gradTerm0_g = columns[7]
    gradTerm1_g = columns[8]
    gradTerm2_g = columns[9]
    for g in range(nGenes):
        l0_g[g] = ltqs_gi[g, 0]
        l1_g[g] = ltqs_gi[g, 1]
        l2_g[g] = ltqs_gi[g, 2]
        w0_g[g] = ltqsVars_gi[g, 0]
        w1_g[g] = ltqsVars_gi[g, 1]
        w2_g[g] = ltqsVars_gi[g, 2]
    invalid = False
    for g in range(nGenes):
        l0 = l0_g[g]
        l1 = l1_g[g]
        l2 = l2_g[g]
        w0 = 1. / (w0_g[g] + t0)
        w1 = 1. / (w1_g[g] + t1)
        w2 = 1. / (w2_g[g] + t2)
        W = ((0. + w0) + w1) + w2
        r0 = w0 / W
        r1 = w1 / W
        r2 = w2 / W
        xr = ((0. + r0 * l0) + r1 * l1) + r2 * l2
        d0 = xr - l0
        d1 = xr - l1
        d2 = xr - l2
        q0 = w0 * (d0 * d0)
        q1 = w1 * (d1 * d1)
        q2 = w2 * (d2 * d2)
        sq = ((0. + q0) + q1) + q2
        gradTerm0_g[g] = w0 * ((q0 - 1.) + r0)
        gradTerm1_g[g] = w1 * ((q1 - 1.) + r1)
        gradTerm2_g[g] = w2 * ((q2 - 1.) + r2)
        w0_g[g] = w0
        w1_g[g] = w1
        w2_g[g] = w2
        W_g[g] = W
        sqDists_g[g] = sq
        invalid |= (_notPositive(w0) | _notPositive(w1) | _notPositive(w2) | _notFinite(W) | _notFinite(xr)
                    | _notFinite(sq))
    grad0 = 0.
    grad1 = 0.
    grad2 = 0.
    for g in range(nGenes):
        grad0 += gradTerm0_g[g]
        grad1 += gradTerm1_g[g]
        grad2 += gradTerm2_g[g]
        logBuffer_g4[g, 0] = w0_g[g]
        logBuffer_g4[g, 1] = w1_g[g]
        logBuffer_g4[g, 2] = w2_g[g]
        logBuffer_g4[g, 3] = W_g[g]
    if sequential:
        negGrad[0] = -(grad0 - grad1)
        negGrad[1] = -(t2 * grad2)
    else:
        negGrad[0] = -t0 * grad0
        negGrad[1] = -t1 * grad1
        negGrad[2] = -t2 * grad2
    for i in range(negGrad.shape[0]):
        invalid |= _notFinite(negGrad[i])
    return FALLBACK if invalid else DONE


@_kernel(_float(_outMatrix, _outVector))
def starTreeLoglik(logs_gk, sqDists_g):
    """np.sum(np.sum(np.log(wbar_gi), axis=1) - np.log(W_g) - np.sum(sqdistsWbar_gi, axis=1)) from logs_gk, the logs
    of [wbar_gi | W_g], and the summed squared distances; sqDists_g is overwritten."""
    nGenes, nColumns = logs_gk.shape
    if sqDists_g.shape[0] != nGenes:
        return np.nan
    stack = _newSumStack()
    logWbarSums_g = np.empty(nGenes)
    _rowSums(logs_gk, nColumns - 1, logWbarSums_g, stack)
    for g in range(nGenes):
        sqDists_g[g] = (logWbarSums_g[g] - logs_gk[g, nColumns - 1]) - sqDists_g[g]
    return _npSum(sqDists_g, nGenes, stack)


@_kernel(_int(_vector, _vector, _vector, _vector, _vector, _vector, _vector, _outVector, _outMatrix, _outMatrix))
def pairStarTree(rootMinusFirst_W_g, rootMinusFirst_ltqs, ltqs1, ltqsVars1, ltqs2, ltqsVars2, wbar2_g, WR_g,
                 ltqs_gi, ltqsVars_gi):
    """calcSingleDLogL's three-leaf star tree: WR_g = rootMinusFirst_W_g - wbar2_g, and the columns
    (ltqs1, ltqs2, (rootMinusFirst_ltqs - wbar2_g * ltqs2) / WR_g) and (ltqsVars1, ltqsVars2, 1 / WR_g)."""
    nGenes = rootMinusFirst_W_g.shape[0]
    if (rootMinusFirst_ltqs.shape[0] != nGenes or ltqs1.shape[0] != nGenes or ltqsVars1.shape[0] != nGenes
            or ltqs2.shape[0] != nGenes or ltqsVars2.shape[0] != nGenes or wbar2_g.shape[0] != nGenes
            or WR_g.shape[0] != nGenes or ltqs_gi.shape[0] != nGenes or ltqs_gi.shape[1] != 3
            or ltqsVars_gi.shape[0] != nGenes or ltqsVars_gi.shape[1] != 3):
        return FALLBACK
    notFinite = False
    tiny = False
    for g in range(nGenes):
        WR = rootMinusFirst_W_g[g] - wbar2_g[g]
        product = wbar2_g[g] * ltqs2[g]
        ltqsR = (rootMinusFirst_ltqs[g] - product) / WR
        ltqsVarsR = 1. / WR
        WR_g[g] = WR
        ltqs_gi[g, 0] = ltqs1[g]
        ltqs_gi[g, 1] = ltqs2[g]
        ltqs_gi[g, 2] = ltqsR
        ltqsVars_gi[g, 0] = ltqsVars1[g]
        ltqsVars_gi[g, 1] = ltqsVars2[g]
        ltqsVars_gi[g, 2] = ltqsVarsR
        notFinite |= _notFinite(WR) | _notFinite(ltqsR) | _notFinite(ltqsVarsR)
        tiny |= _isTiny(product) | _isTiny(ltqsR) | _isTiny(ltqsVarsR)
    if notFinite:
        return FALLBACK
    return DONE | _tinyFlag(tiny)


@_kernel(_int(_vector, _vector, _matrix, _vector, _vector, _vector, _outMatrix, _outMatrix, _outVector))
def pairStarTreeLoglikTerms(xr_g, W_g, ltqs_gi, wbar1_g, wbar2_g, WR_g, wbar_gi, logBuffer_g4, sqDists_g):
    """calcSingleDLogL's closing getLoglikAndGradStarTree(ltqs_gi=..., xr_g=..., wbar_gi=..., W_g=...): wbar_gi =
    (wbar1_g, wbar2_g, WR_g) as columns, [wbar1 | wbar2 | WR | W] into logBuffer_g4 for np.log, and
    sum_i wbar_i (xr - l_i) ** 2 into sqDists_g."""
    nGenes = xr_g.shape[0]
    if (W_g.shape[0] != nGenes or ltqs_gi.shape[0] != nGenes or ltqs_gi.shape[1] != 3 or wbar1_g.shape[0] != nGenes
            or wbar2_g.shape[0] != nGenes or WR_g.shape[0] != nGenes or wbar_gi.shape[0] != nGenes
            or wbar_gi.shape[1] != 3 or logBuffer_g4.shape[0] != nGenes or logBuffer_g4.shape[1] != 4
            or sqDists_g.shape[0] != nGenes):
        return FALLBACK
    invalid = False
    tiny = False
    for g in range(nGenes):
        xr = xr_g[g]
        w0 = wbar1_g[g]
        w1 = wbar2_g[g]
        w2 = WR_g[g]
        d0 = xr - ltqs_gi[g, 0]
        d1 = xr - ltqs_gi[g, 1]
        d2 = xr - ltqs_gi[g, 2]
        s0 = d0 * d0
        s1 = d1 * d1
        s2 = d2 * d2
        q0 = w0 * s0
        q1 = w1 * s1
        q2 = w2 * s2
        sq = ((0. + q0) + q1) + q2
        wbar_gi[g, 0] = w0
        wbar_gi[g, 1] = w1
        wbar_gi[g, 2] = w2
        logBuffer_g4[g, 0] = w0
        logBuffer_g4[g, 1] = w1
        logBuffer_g4[g, 2] = w2
        logBuffer_g4[g, 3] = W_g[g]
        sqDists_g[g] = sq
        invalid |= (_notPositive(w0) | _notPositive(w1) | _notPositive(w2) | _notPositive(W_g[g])
                    | _notFinite(sq))
        tiny |= _isTiny(s0) | _isTiny(s1) | _isTiny(s2) | _isTiny(q0) | _isTiny(q1) | _isTiny(q2)
    if invalid:
        return FALLBACK
    return DONE | _tinyFlag(tiny)


@_kernel(_int(_vector, _vector, _matrix, _matrix, _matrix, _float, _float, _float, _outVector, _outVector))
def derBasedDLogLUBTerms(xr_g, Wr_g, ltqs_gi, ltqsVars_gi, wir_gi, t1, t2, tR, dLdWrTimesWr_g, dLdxrOverSqrtWr_g):
    """estimateDerBasedDLogLUB after its loglik: the vectors dLdWr * Wr_g and dLdxr / np.sqrt(Wr_g) whose norms it
    returns, in its order of operations. It sets no TINY flag: callers use it only while np.geterr()['under'] is
    'ignore'."""
    nGenes = xr_g.shape[0]
    if (Wr_g.shape[0] != nGenes or ltqs_gi.shape[0] != nGenes or ltqs_gi.shape[1] != 3
            or ltqsVars_gi.shape[0] != nGenes or ltqsVars_gi.shape[1] != 3 or wir_gi.shape[0] != nGenes
            or wir_gi.shape[1] != 3 or dLdWrTimesWr_g.shape[0] != nGenes or dLdxrOverSqrtWr_g.shape[0] != nGenes):
        return FALLBACK
    notFinite = False
    for g in range(nGenes):
        # Values numpy computes only as divisors, or not at all, are checked too: numpy warns about those as well
        tbar1 = ltqsVars_gi[g, 0] + t1
        tbar2 = ltqsVars_gi[g, 1] + t2
        tbarR = ltqsVars_gi[g, 2] + tR
        w1a = 1. / tbar1
        w2a = 1. / tbar2
        wRa = 1. / tbarR
        Wa = ((0. + w1a) + w2a) + wRa
        w1aOverW = w1a / Wa
        w2aOverW = w2a / Wa
        wRaOverW = wRa / Wa
        w1r = wir_gi[g, 0]
        w2r = wir_gi[g, 1]
        wRr = wir_gi[g, 2]
        ltqs1 = ltqs_gi[g, 0]
        ltqs2 = ltqs_gi[g, 1]
        ltqsR = ltqs_gi[g, 2]
        Wr = Wr_g[g]

        diff12 = ltqs1 - ltqs2
        diff1R = ltqs1 - ltqsR
        diff2R = ltqs2 - ltqsR
        diffrR = xr_g[g] - ltqsR
        v12 = diff12 * diff12
        v1R = diff1R * diff1R
        v2R = diff2R * diff2R

        da = w1a * w2a * v12 + w1a * wRa * v1R + w2a * wRa * v2R
        dr = w1r * w2r * v12 + w1r * wRr * v1R + w2r * wRr * v2R
        wRrOverW = wRr / Wr
        wRrSquared = wRr * wRr
        dLdWr = (wRa * (1 - wRaOverW) - wRr * (1 - wRrOverW) +
                 wRaOverW * (w1a * w2a * v12 - (1 - wRaOverW) * da) -
                 wRrOverW * (w1r * w2r * v12 - (1 - wRrOverW) * dr)) / wRrSquared + \
            (2 * diffrR / wRr) * (
                wRaOverW * (w1a * diff1R + w2a * diff2R) - wRrOverW * (w1r * diff1R + w2r * diff2R))
        dLdxr = 2 * (Wr / wRr) * (diff1R * (w1a * wRaOverW - w1r * wRrOverW)
                                  + diff2R * (w2a * wRaOverW - w2r * wRrOverW))
        sqrtWr = np.sqrt(Wr)
        dLdWrTimesWr = dLdWr * Wr
        dLdxrOverSqrtWr = dLdxr / sqrtWr
        dLdWrTimesWr_g[g] = dLdWrTimesWr
        dLdxrOverSqrtWr_g[g] = dLdxrOverSqrtWr
        notFinite |= (_notFinite(tbar1) | _notFinite(tbar2) | _notFinite(tbarR) | _notFinite(Wa)
                      | _notFinite(w1aOverW) | _notFinite(w2aOverW) | _notFinite(wRrSquared) | _notFinite(sqrtWr)
                      | _notFinite(dLdWrTimesWr) | _notFinite(dLdxrOverSqrtWr))
    return FALLBACK if notFinite else DONE


# Moving the root along an edge


@_kernel(_int(_vector, _vector, _float, _vector, _vector, _outVector, _outVector))
def ltqsAsIfRoot(nodeLtqs_g, nodeW_g, tConn, rootLtqs_g, rootW_g, ltqsAIRoot_g, WAIRoot_g):
    """getLtqsAsIfRoot. Returns AS_NODE where it returns copies of the node's own ltqs and W."""
    nGenes = nodeLtqs_g.shape[0]
    if (nGenes == 0 or nodeW_g.shape[0] != nGenes or rootLtqs_g.shape[0] != nGenes or rootW_g.shape[0] != nGenes
            or ltqsAIRoot_g.shape[0] != nGenes or WAIRoot_g.shape[0] != nGenes):
        return FALLBACK
    notFinite = False
    tiny = False
    maxRootMinusNodeW = -np.inf
    # First pass: wbarNode into ltqsAIRoot_g and rootMinusNodeW into WAIRoot_g
    for g in range(nGenes):
        nodeLtqsVar = 1. / nodeW_g[g]
        tbarNode = tConn + nodeLtqsVar
        wbarNode = 1. / tbarNode
        rootMinusNodeW = rootW_g[g] - wbarNode
        ltqsAIRoot_g[g] = wbarNode
        WAIRoot_g[g] = rootMinusNodeW
        maxRootMinusNodeW = max(maxRootMinusNodeW, rootMinusNodeW)
        notFinite |= _notFinite(nodeLtqsVar) | _notFinite(tbarNode) | _notFinite(wbarNode) | _notFinite(rootMinusNodeW)
        tiny |= _isTiny(nodeLtqsVar) | _isTiny(wbarNode)
    if notFinite:
        return FALLBACK
    if maxRootMinusNodeW < 1e-12:
        return AS_NODE | _tinyFlag(tiny)
    for g in range(nGenes):
        wbarNode = ltqsAIRoot_g[g]
        rootMinusNodeW = WAIRoot_g[g]
        rootProduct = rootLtqs_g[g] * rootW_g[g]
        nodeProduct = wbarNode * nodeLtqs_g[g]
        rootMinusNodeLtqs = (rootProduct - nodeProduct) / rootMinusNodeW
        rootMinusNodeLtqsVar = 1. / rootMinusNodeW
        tbarRoot = tConn + rootMinusNodeLtqsVar
        wbarRoot = 1. / tbarRoot
        nodePlusRootW = nodeW_g[g] + wbarRoot
        nodeLtqsTimesW = nodeLtqs_g[g] * nodeW_g[g]
        rootLtqsTimesW = wbarRoot * rootMinusNodeLtqs
        nodePlusRootLtqs = (nodeLtqsTimesW + rootLtqsTimesW) / nodePlusRootW
        ltqsAIRoot_g[g] = nodePlusRootLtqs
        WAIRoot_g[g] = nodePlusRootW
        notFinite |= (_notFinite(rootProduct) | _notFinite(nodeProduct) | _notFinite(rootMinusNodeLtqs)
                      | _notFinite(rootMinusNodeLtqsVar) | _notFinite(tbarRoot) | _notFinite(wbarRoot)
                      | _notFinite(nodePlusRootW) | _notFinite(nodeLtqsTimesW) | _notFinite(rootLtqsTimesW)
                      | _notFinite(nodePlusRootLtqs))
        tiny |= (_isTiny(rootProduct) | _isTiny(nodeProduct) | _isTiny(rootMinusNodeLtqs)
                 | _isTiny(rootMinusNodeLtqsVar) | _isTiny(wbarRoot) | _isTiny(nodeLtqsTimesW)
                 | _isTiny(rootLtqsTimesW) | _isTiny(nodePlusRootLtqs))
    if notFinite:
        return FALLBACK
    return DONE | _tinyFlag(tiny)


# A node's position given its children


@_kernel(_int(_vector, _outVector))
def reciprocal(W_g, ltqsVars_g):
    """1 / W_g, the conversion TreeNode.getLtqsVars makes of a cached precision."""
    nGenes = W_g.shape[0]
    if ltqsVars_g.shape[0] != nGenes:
        return FALLBACK
    notFinite = False
    tiny = False
    for g in range(nGenes):
        ltqsVar = 1. / W_g[g]
        ltqsVars_g[g] = ltqsVar
        notFinite |= _notFinite(ltqsVar)
        tiny |= _isTiny(ltqsVar)
    if notFinite:
        return FALLBACK
    return DONE | _tinyFlag(tiny)


MAX_NODE_STEP_CHILDREN = 4


@_kernel(_int(types.UniTuple(_vector, MAX_NODE_STEP_CHILDREN), types.UniTuple(_vector, MAX_NODE_STEP_CHILDREN),
              types.UniTuple(_float, MAX_NODE_STEP_CHILDREN), _int, types.boolean, _outVector, _outVector, _outMatrix,
              _outVector))
def nodeStep(ltqs_c, ltqsVars_c, tParent_c, nChildren, withLoglikTerms, xr_g, W_g, logBuffer_cg, sqDists_c):
    """findNodeLtqsGivenLeafs(childNodes=...) for the first nChildren children, 1 to 4 (the tuples' other entries are
    ignored): per gene W_g = (0 + w1) + w2 ... and xr_g = ((0 + l1 w1) + l2 w2 ...) / W_g with
    w = 1 / (ltqsVars + tParent), child by child. With withLoglikTerms, also the terms of
    getLoglikAndGradStarTree(childNodes=..., xr_g=xr_g, W_g=W_g): [W_g; w1; w2; ...] into logBuffer_cg for np.log and
    each child's np.sum(w (xr_g - l) ** 2) into sqDists_c."""
    nGenes = W_g.shape[0]
    if not 1 <= nChildren <= MAX_NODE_STEP_CHILDREN:
        return FALLBACK
    if xr_g.shape[0] != nGenes:
        return FALLBACK
    for c in range(nChildren):
        if ltqs_c[c].shape[0] != nGenes or ltqsVars_c[c].shape[0] != nGenes:
            return FALLBACK
    if withLoglikTerms and (logBuffer_cg.shape[0] != nChildren + 1 or logBuffer_cg.shape[1] != nGenes
                            or sqDists_c.shape[0] != nChildren):
        return FALLBACK

    for g in range(nGenes):
        W_g[g] = 0.
        xr_g[g] = 0.
    notFinite = False
    notPositive = False
    tiny = False
    for c in range(nChildren):
        ltqs_g = ltqs_c[c]
        ltqsVars_g = ltqsVars_c[c]
        tParent = tParent_c[c]
        for g in range(nGenes):
            wbar = 1. / (ltqsVars_g[g] + tParent)
            product = ltqs_g[g] * wbar
            W_g[g] += wbar
            xr_g[g] += product
            notFinite |= _notFinite(wbar)
            notPositive |= wbar <= 0.
            tiny |= _isTiny(wbar) | _isTiny(product)
    for g in range(nGenes):
        xr = xr_g[g] / W_g[g]
        xr_g[g] = xr
        notFinite |= _notFinite(W_g[g]) | _notFinite(xr)
        tiny |= _isTiny(xr)
    if notFinite:
        return FALLBACK
    if not withLoglikTerms:
        return DONE | _tinyFlag(tiny)
    if notPositive:
        return FALLBACK

    stack = _newSumStack()
    terms_g = np.empty(nGenes)
    for g in range(nGenes):
        logBuffer_cg[0, g] = W_g[g]
    for c in range(nChildren):
        ltqs_g = ltqs_c[c]
        ltqsVars_g = ltqsVars_c[c]
        tParent = tParent_c[c]
        wbar_g = logBuffer_cg[c + 1]
        for g in range(nGenes):
            wbar = 1. / (ltqsVars_g[g] + tParent)
            diff = xr_g[g] - ltqs_g[g]
            sqDist = diff * diff
            term = wbar * sqDist
            wbar_g[g] = wbar
            terms_g[g] = term
            tiny |= _isTiny(sqDist) | _isTiny(term)
        sqDists_c[c] = _npSum(terms_g, nGenes, stack)
        notFinite |= _notFinite(sqDists_c[c])
    if notFinite:
        return FALLBACK
    return DONE | _tinyFlag(tiny)


@_kernel(_float(_outMatrix, _outVector))
def nodeLoglik(logs_cg, sqDists_c):
    """getLoglikAndGradStarTree(childNodes=..., xr_g=..., W_g=...): -np.sum(np.log(W_g)), then per child
    + np.sum(np.log(wbar_g)) - np.sum(wbar_g * (xr_g - ltqs) ** 2), from the logs of nodeStep's buffer."""
    nGenes = logs_cg.shape[1]
    if logs_cg.shape[0] != sqDists_c.shape[0] + 1:
        return np.nan
    stack = _newSumStack()
    loglik = -_npSum(logs_cg[0], nGenes, stack)
    for c in range(sqDists_c.shape[0]):
        loglik += _npSum(logs_cg[c + 1], nGenes, stack)
        loglik -= sqDists_c[c]
    return loglik


@_kernel(_int(_matrix, _matrix, _vector, _outVector, _outVector, _outMatrix, _outVector))
def nodeStarTerms(ltqs_gk, ltqsVars_gk, t_k, xr_g, W_g, logBuffer_gk1, sqDists_g):
    """findNodeLtqsGivenLeafs(ltqs_gi=..., return_wbar_gi=True) and the terms of getLoglikAndGradStarTree(ltqs_gi=...,
    xr_g=..., wbar_gi=..., W_g=...) for any number of children: xr_g, W_g, [wbar_gi | W_g] into logBuffer_gk1 for
    np.log and np.sum(wbar_gi * (xr_g[:, None] - ltqs_gi) ** 2, axis=1) into sqDists_g."""
    nGenes, nChildren = ltqs_gk.shape
    if (nGenes == 0 or nChildren == 0 or ltqsVars_gk.shape[0] != nGenes or ltqsVars_gk.shape[1] != nChildren
            or t_k.shape[0] != nChildren or xr_g.shape[0] != nGenes or W_g.shape[0] != nGenes
            or logBuffer_gk1.shape[0] != nGenes or logBuffer_gk1.shape[1] != nChildren + 1
            or sqDists_g.shape[0] != nGenes):
        return FALLBACK
    stack = _newSumStack()
    terms_gk = np.empty((nGenes, nChildren))
    invalid = False
    tiny = False
    for g in range(nGenes):
        for c in range(nChildren):
            wbar = 1. / (ltqsVars_gk[g, c] + t_k[c])
            logBuffer_gk1[g, c] = wbar
            invalid |= _notPositive(wbar)
            tiny |= _isTiny(wbar)
    _rowSums(logBuffer_gk1, nChildren, W_g, stack)
    for g in range(nGenes):
        W = W_g[g]
        logBuffer_gk1[g, nChildren] = W
        for c in range(nChildren):
            wOverW = logBuffer_gk1[g, c] / W
            product = wOverW * ltqs_gk[g, c]
            terms_gk[g, c] = product
            tiny |= _isTiny(wOverW) | _isTiny(product)
        invalid |= _notFinite(W)
    _rowSums(terms_gk, nChildren, xr_g, stack)
    for g in range(nGenes):
        xr = xr_g[g]
        for c in range(nChildren):
            diff = xr - ltqs_gk[g, c]
            sqDist = diff * diff
            term = logBuffer_gk1[g, c] * sqDist
            terms_gk[g, c] = term
            tiny |= _isTiny(sqDist) | _isTiny(term)
        invalid |= _notFinite(xr)
    _rowSums(terms_gk, nChildren, sqDists_g, stack)
    for g in range(nGenes):
        invalid |= _notFinite(sqDists_g[g])
    if invalid:
        return FALLBACK
    return DONE | _tinyFlag(tiny)


# The objective of Tree.optTimes, on a table of the tree


@_kernel(_int(_vector, _indices, _indices, _indices, _indices, _outMatrix, _outMatrix, _outMatrix, _outVector,
              _outMatrix, _outMatrix, _outVector))
def treeTerms(t_e, childStart_j, childRow_e, internalRow_j, rowInternal_r, ltqs_rg, ltqsVars_rg, logBuffer_eg,
              sqDists_e, xrAsIfRoot_jg, WAsIfRoot_jg, der_e):
    """Tree.getLikelihoodGradGivenTimes with mem_friendly=True, on a table of the tree. Rows r of ltqs_rg and
    ltqsVars_rg are the nodes in pre-order (row 0 the root); edge e = r - 1 leads to row r, with time t_e[e].
    Internal nodes j (in pre-order, j = 0 the root) sit at rows internalRow_j[j], with their children's rows at
    childRow_e[childStart_j[j]:childStart_j[j + 1]]; rowInternal_r maps rows back to j, -1 for leafs.

    Up-pass, as TreeNode.getLtqsComplete: each internal node's xr (into ltqs_rg) and, below the root, 1 / W as its
    parent's getLtqsVars(mem_friendly=True) reads it (into ltqsVars_rg); every edge's wbar into logBuffer_eg[e] and
    every internal node's W into logBuffer_eg[nEdges + j], for np.log; every edge's np.sum(wbar * (xr - ltqs) ** 2)
    into sqDists_e. Down-pass, as TreeNode.getDerivativesDownstream: every edge's derivative into der_e, with each
    internal node's position as if it were the root in xrAsIfRoot_jg and WAsIfRoot_jg. It sets no TINY flag: callers
    use it only while np.geterr()['under'] is 'ignore'."""
    nEdges = t_e.shape[0]
    nInternal = internalRow_j.shape[0]
    nGenes = ltqs_rg.shape[1]
    if (nInternal == 0 or childStart_j.shape[0] != nInternal + 1 or childRow_e.shape[0] != nEdges
            or rowInternal_r.shape[0] != nEdges + 1 or ltqs_rg.shape[0] != nEdges + 1
            or ltqsVars_rg.shape[0] != nEdges + 1 or ltqsVars_rg.shape[1] != nGenes
            or logBuffer_eg.shape[0] != nEdges + nInternal or logBuffer_eg.shape[1] != nGenes
            or sqDists_e.shape[0] != nEdges or xrAsIfRoot_jg.shape[0] != nInternal
            or xrAsIfRoot_jg.shape[1] != nGenes or WAsIfRoot_jg.shape[0] != nInternal
            or WAsIfRoot_jg.shape[1] != nGenes or der_e.shape[0] != nEdges):
        return FALLBACK
    stack = _newSumStack()
    terms_g = np.empty(nGenes)
    invalid = False

    for j in range(nInternal - 1, -1, -1):
        xr_g = ltqs_rg[internalRow_j[j]]
        W_g = logBuffer_eg[nEdges + j]
        for g in range(nGenes):
            W_g[g] = 0.
            xr_g[g] = 0.
        for i in range(childStart_j[j], childStart_j[j + 1]):
            child = childRow_e[i]
            tParent = t_e[child - 1]
            ltqs_g = ltqs_rg[child]
            ltqsVars_g = ltqsVars_rg[child]
            wbar_g = logBuffer_eg[child - 1]
            for g in range(nGenes):
                wbar = 1. / (ltqsVars_g[g] + tParent)
                wbar_g[g] = wbar
                W_g[g] += wbar
                xr_g[g] += ltqs_g[g] * wbar
                invalid |= _notPositive(wbar)
        for g in range(nGenes):
            xr = xr_g[g] / W_g[g]
            xr_g[g] = xr
            invalid |= _notFinite(W_g[g]) | _notFinite(xr)
        if j > 0:
            ltqsVars_g = ltqsVars_rg[internalRow_j[j]]
            for g in range(nGenes):
                ltqsVar = 1. / W_g[g]
                ltqsVars_g[g] = ltqsVar
                invalid |= _notFinite(ltqsVar)
        for i in range(childStart_j[j], childStart_j[j + 1]):
            child = childRow_e[i]
            ltqs_g = ltqs_rg[child]
            wbar_g = logBuffer_eg[child - 1]
            for g in range(nGenes):
                diff = xr_g[g] - ltqs_g[g]
                terms_g[g] = wbar_g[g] * (diff * diff)
            sqDists = _npSum(terms_g, nGenes, stack)
            sqDists_e[child - 1] = sqDists
            invalid |= _notFinite(sqDists)
        if invalid:
            return FALLBACK

    rootRow = internalRow_j[0]
    for g in range(nGenes):
        xrAsIfRoot_jg[0, g] = ltqs_rg[rootRow, g]
        WAsIfRoot_jg[0, g] = logBuffer_eg[nEdges, g]
    ltqsTimesW_g = np.empty(nGenes)
    for j in range(nInternal):
        xrAsIfRoot_g = xrAsIfRoot_jg[j]
        WAsIfRoot_g = WAsIfRoot_jg[j]
        for g in range(nGenes):
            ltqsTimesW_g[g] = xrAsIfRoot_g[g] * WAsIfRoot_g[g]
        for i in range(childStart_j[j], childStart_j[j + 1]):
            child = childRow_e[i]
            tParent = t_e[child - 1]
            ltqs_g = ltqs_rg[child]
            ltqsVars_g = ltqsVars_rg[child]
            childInternal = rowInternal_r[child]
            for g in range(nGenes):
                wbarChild = 1. / (tParent + ltqsVars_g[g])
                WWOChild = WAsIfRoot_g[g] - wbarChild
                ltqsWOChild = (ltqsTimesW_g[g] - wbarChild * ltqs_g[g]) / WWOChild
                ltqsVarsWOChild = 1. / WWOChild
                diff = ltqsWOChild - ltqs_g[g]
                sqDist = diff * diff
                totVar = tParent + (ltqsVars_g[g] + ltqsVarsWOChild)
                terms_g[g] = (-1. + sqDist / totVar) / totVar
                invalid |= _notFinite(ltqsWOChild) | _notFinite(ltqsVarsWOChild) | _notFinite(totVar)
                if childInternal >= 0:
                    wbarRoot = 1. / (tParent + ltqsVarsWOChild)
                    WChild = 1. / ltqsVars_g[g]
                    WChildWithRoot = WChild + wbarRoot
                    ltqsChildWithRoot = (ltqs_g[g] * WChild + wbarRoot * ltqsWOChild) / WChildWithRoot
                    xrAsIfRoot_jg[childInternal, g] = ltqsChildWithRoot
                    WAsIfRoot_jg[childInternal, g] = WChildWithRoot
                    invalid |= (_notFinite(wbarRoot) | _notFinite(WChild) | _notFinite(WChildWithRoot)
                                | _notFinite(ltqsChildWithRoot))
            der = _npSum(terms_g, nGenes, stack)
            der_e[child - 1] = der
            invalid |= _notFinite(der)
        if invalid:
            return FALLBACK
    return DONE


@_kernel(_float(_outMatrix, _outVector, _indices, _indices, _indices))
def treeLoglik(logs_eg, sqDists_e, childStart_j, childRow_e, rowInternal_r):
    """The root's prefactor from treeTerms' results, in TreeNode.getLtqsComplete's order of additions: each node's
    children's prefactors from 0, plus -np.sum(np.log(W_g)), then per child + np.sum(np.log(wbar_g)) and
    - np.sum(wbar_g * (xr_g - ltqs) ** 2)."""
    nEdges = sqDists_e.shape[0]
    nInternal = childStart_j.shape[0] - 1
    nGenes = logs_eg.shape[1]
    if logs_eg.shape[0] != nEdges + nInternal or childRow_e.shape[0] != nEdges or rowInternal_r.shape[0] != nEdges + 1:
        return np.nan
    stack = _newSumStack()
    prefactor_j = np.empty(nInternal)
    for j in range(nInternal - 1, -1, -1):
        prefactor = 0.
        for i in range(childStart_j[j], childStart_j[j + 1]):
            childInternal = rowInternal_r[childRow_e[i]]
            prefactor += prefactor_j[childInternal] if childInternal >= 0 else 0.
        loglik = -_npSum(logs_eg[nEdges + j], nGenes, stack)
        for i in range(childStart_j[j], childStart_j[j + 1]):
            edge = childRow_e[i] - 1
            loglik += _npSum(logs_eg[edge], nGenes, stack)
            loglik -= sqDists_e[edge]
        prefactor_j[j] = prefactor + loglik
    return prefactor_j[0]
