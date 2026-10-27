"""Success-rate statistics for policy evaluation (lesson 25 · Build it).

A trial ends in 1 or 0; k successes in n trials. Intervals, exact coverage by binomial sums, trial budgets,
and Fisher's exact test, in plain Python/NumPy. Later lessons report k/n with `wilson`.
"""
import math


def wald(k, n, z=1.96):
    """Textbook normal-approximation interval p +- z sqrt(p(1-p)/n), clipped to [0, 1]."""
    p = k / n
    h = z * math.sqrt(p * (1 - p) / n)
    return max(0.0, p - h), min(1.0, p + h)


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials (Wilson 1927)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def binom_pmf(k, n, p):
    """Probability of exactly k successes in n independent trials with rate p."""
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def coverage(interval, n, p, z=1.96):
    """Exact probability that interval(k, n) contains p when k ~ Binomial(n, p)."""
    return sum(binom_pmf(k, n, p) for k in range(n + 1) if interval(k, n, z)[0] <= p <= interval(k, n, z)[1])


def trials_needed(p, half_width, z=1.96):
    """Smallest n whose Wilson interval at observed rate p has half-width <= half_width."""
    n = 1
    while True:
        lo, hi = wilson(round(p * n), n, z)
        if (hi - lo) / 2 <= half_width:
            return n
        n += 1


def fisher_p(k1, n1, k2, n2):
    """Two-sided Fisher exact test for k1/n1 vs k2/n2: the total probability, with the margins fixed,
    of all tables no more likely than the observed one."""
    K, N = k1 + k2, n1 + n2
    pmf = {a: math.comb(n1, a) * math.comb(n2, K - a) / math.comb(N, K) for a in range(max(0, K - n2), min(K, n1) + 1)}
    return min(1.0, sum(q for q in pmf.values() if q <= pmf[k1] * (1 + 1e-7)))


def power(p1, p2, n, alpha=0.05):
    """Exact probability that Fisher's test at level alpha separates two policies with rates p1, p2, n trials each."""
    pm1 = [binom_pmf(k, n, p1) for k in range(n + 1)]
    pm2 = [binom_pmf(k, n, p2) for k in range(n + 1)]
    return sum(pm1[a] * pm2[b] for a in range(n + 1) for b in range(n + 1) if fisher_p(a, n, b, n) < alpha)
