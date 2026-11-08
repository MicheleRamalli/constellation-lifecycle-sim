"""Analytical reference models used for validation and for the figures.

- Renewal-reward availability of one slot (launch on failure).
- Birth-death Markov chain of a pool of identical units with independent
  failure and replacement (in-orbit spares, one reorder per failure).
- Binomial distribution of the operational units of the same pool.

All times in seconds, all rates in 1/s.
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import binom

from reliability import reliability_analytic

YEAR_S = 365.25 * 86400.0  # only used to rescale the integration variable


def slot_availability_renewal_reward(reliability, design_life_s, order_to_launch_s,
                                     commissioning_s, delay_s=0.0):
    """Long-run fraction of time a slot is operational (renewal-reward theorem).

    A cycle goes from one launch to the next: life L = min(T_fail, design life)
    counted from launch, then a detection delay, then the order-to-launch time.
    The slot is up only after commissioning: up = max(L - commissioning, 0).
    Using E[X] = integral of P(X > t) dt:
        E[cycle] = order_to_launch + delay + integral_0^DL R(t) dt
        E[up]    = integral_comm^DL R(t) dt + delay
    A = E[up] / E[cycle].
    delay_s is used only to bound the effect of the time-step discretization:
    on the grid both the up time and the cycle grow by a delay in [0, dt).

    Inputs:
        reliability       : dict with the reliability model (see lifecycle.launch_satellites)
        design_life_s     : design life [s] (np.inf = no retirement)
        order_to_launch_s : delay from order to launch [s]
        commissioning_s   : delay from launch to operational [s]
        delay_s           : extra delay added to each life [s]
    Output:
        availability A [-]
    """
    # quad works on the variable t / YEAR (order 1): integrating directly in
    # seconds up to np.inf is badly scaled and quad returns ~0.
    def R_of_years(t_years):
        return reliability_analytic(t_years * YEAR_S, **reliability)

    e_life_s = YEAR_S * quad(R_of_years, 0.0, design_life_s / YEAR_S, limit=200)[0]
    e_up_s = YEAR_S * quad(R_of_years, commissioning_s / YEAR_S, design_life_s / YEAR_S,
                           limit=200)[0]
    return (e_up_s + delay_s) / (order_to_launch_s + e_life_s + delay_s)


def slot_availability_pre_order(reliability, design_life_s, order_to_launch_s,
                                commissioning_s, delay_s=0.0):
    """Long-run fraction of time a slot is operational with early ordering (strategy c).

    Each satellite orders its successor at age a = DL - order - commissioning.
    A cycle goes from one launch to the next:
        T < a  (lost before the planned order): cycle = T + order (+ detection delay)
        T >= a (planned successor)            : cycle = DL - commissioning
    E[cycle] = E[T 1{T<a}] + order F(a) + (DL - comm) R(a)
             = integral_0^a R dt - a R(a) + order (1 - R(a)) + (DL - comm) R(a)
             = order + integral_0^a R(t) dt          (since DL - comm - a = order)
    Each satellite is usable from launch + commissioning to min(T, DL), and the
    successor starts exactly at DL, so E[up] = integral_comm^DL R(t) dt as in a).
    A = E[up] / E[cycle]. Without failures (R = 1): A = (DL - comm) / (DL - comm) = 1.
    delay_s bounds the grid effect as in slot_availability_renewal_reward.

    Inputs/Output: as slot_availability_renewal_reward.
    """
    def R_of_years(t_years):
        return reliability_analytic(t_years * YEAR_S, **reliability)

    pre_order_age_s = design_life_s - order_to_launch_s - commissioning_s
    e_cycle_s = order_to_launch_s + YEAR_S * quad(R_of_years, 0.0, pre_order_age_s / YEAR_S,
                                                  limit=200)[0]
    e_up_s = YEAR_S * quad(R_of_years, commissioning_s / YEAR_S, design_life_s / YEAR_S,
                           limit=200)[0]
    return (e_up_s + delay_s) / (e_cycle_s + delay_s)


def birth_death_stationary(n_units, fail_rate, repair_rate):
    """Stationary distribution of the number of units down in a finite pool.

    Continuous-time Markov chain. State k = number of units down (0..n_units).
    Each working unit fails with rate fail_rate; each unit down is replaced
    independently with rate repair_rate (unlimited servers):
        k -> k+1 with rate fail_rate * (n_units - k)
        k -> k-1 with rate repair_rate * k
    Solves pi Q = 0 with sum(pi) = 1 by a linear system.

    Inputs:
        n_units     : number of units in the pool [-]
        fail_rate   : failure rate of a working unit [1/s]
        repair_rate : replacement rate of a unit down [1/s]
    Output:
        pi : probability of k units down, shape (n_units + 1,)
    """
    n_states = n_units + 1
    Q = np.zeros((n_states, n_states))  # generator matrix
    for k in range(n_states):
        if k < n_units:
            Q[k, k + 1] = fail_rate * (n_units - k)
        if k > 0:
            Q[k, k - 1] = repair_rate * k
        Q[k, k] = -np.sum(Q[k, :])
    # pi Q = 0  <=>  Q^T pi^T = 0; replace the last equation by sum(pi) = 1
    A = Q.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(n_states)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


def active_distribution_from_down(p_down, n_slots):
    """Distribution of operational satellites from the distribution of units down.

    With U usable units out of a pool of n_units, the operational satellites are
    min(n_slots, U): the spares fill every empty slot.

    Inputs:
        p_down  : probability of k units down, k = 0..n_units, shape (n_units + 1,)
        n_slots : satellite positions in the plane [-]
    Output:
        P(n_active = j) for j = 0..n_slots, shape (n_slots + 1,)
    """
    n_units = len(p_down) - 1
    p_usable = p_down[::-1]  # P(U = u) = P(down = n_units - u)
    p_active = p_usable[:n_slots + 1].copy()
    p_active[n_slots] = np.sum(p_usable[n_slots:])  # U >= n_slots fills every slot
    return p_active


def active_distribution_binomial(n_slots, n_spares, p_up):
    """Distribution of operational satellites with in-orbit spares (pool model).

    The pool of n_slots + n_spares units has n_units - U ~ Binomial(n_units, 1 - p_up)
    units down; the operational satellites are min(n_slots, U).

    Inputs:
        n_slots  : satellite positions in the plane [-]
        n_spares : in-orbit spares [-]
        p_up     : long-run probability that a unit is usable [-]
    Output:
        P(n_active = j) for j = 0..n_slots, shape (n_slots + 1,)
    """
    n_units = n_slots + n_spares
    p_down = binom.pmf(np.arange(n_units + 1), n_units, 1.0 - p_up)
    return active_distribution_from_down(p_down, n_slots)
