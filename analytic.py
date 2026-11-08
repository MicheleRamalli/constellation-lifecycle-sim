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


def active_distribution_from_usable(p_usable, n_slots):
    """Distribution of operational satellites from the distribution of usable satellites.

    With U usable satellites (commissioned and alive), the operational ones are
    min(n_slots, U): the spares fill every empty slot.

    Inputs:
        p_usable : probability of U = u, u = 0..u_max, shape (u_max + 1,)
        n_slots  : satellite positions in the plane [-]
    Output:
        P(n_active = j) for j = 0..n_slots, shape (n_slots + 1,)
    """
    p_active = np.zeros(n_slots + 1)
    n_copy = min(len(p_usable), n_slots)
    p_active[:n_copy] = p_usable[:n_copy]
    p_active[n_slots] = np.sum(p_usable[n_slots:])  # U >= n_slots fills every slot
    return p_active


def active_distribution_from_down(p_down, n_slots):
    """Distribution of operational satellites from the distribution of units down.

    Inputs:
        p_down  : probability of k units down, k = 0..n_units, shape (n_units + 1,)
        n_slots : satellite positions in the plane [-]
    Output:
        P(n_active = j) for j = 0..n_slots, shape (n_slots + 1,)
    """
    return active_distribution_from_usable(p_down[::-1], n_slots)  # U = n_units - down


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


def usable_distribution_cadence(t_s, launch_times_s, launch_counts, reliability, design_life_s,
                                commissioning_s, launch_success_prob):
    """Exact distribution of usable satellites at time t, fixed cadence (strategy d).

    No satellite is ever reordered and every satellite is independent (H1, H2:
    same R(t) from launch, active or spare). A satellite of the cohort launched
    at t_j is usable at time t iff its launch succeeded, it is commissioned
    (age >= C), it has not failed (prob. R(age)) and it has not retired
    (age < DL). So each cohort gives Binomial(k_j, q_j), q_j = p_launch R(t - t_j),
    and U(t) is their sum (a Poisson-binomial distribution), computed exactly
    by convolving the binomial pmfs.

    Inputs:
        t_s                 : time [s]
        launch_times_s      : launch time of each cohort [s], shape (n_cohorts,)
        launch_counts       : satellites in each cohort [-], shape (n_cohorts,)
        reliability         : dict with the reliability model
        design_life_s       : design life [s]
        commissioning_s     : commissioning time [s]
        launch_success_prob : probability that a launch succeeds [-]
    Output:
        P(U = u), u = 0..total satellites of the usable cohorts
    """
    pmf = np.array([1.0])
    for t_j, k_j in zip(launch_times_s, launch_counts):
        age_s = t_s - t_j
        if commissioning_s <= age_s < design_life_s:
            q = launch_success_prob * reliability_analytic(age_s, **reliability)
            pmf = np.convolve(pmf, binom.pmf(np.arange(k_j + 1), k_j, q))
    return pmf


def reference_launch_rate(n_slots, reliability, design_life_s, commissioning_s):
    """Reference launch rate N / E[up] [satellites/s].

    E[up] = integral_C^DL R(t) dt is the mean useful life of a satellite
    (usable from commissioning to min(failure, design life)). N / E[up] is the
    rate at which every satellite gives its whole useful life and no slot is
    ever empty (no gaps, no unused spares). It is a reference, not a strict
    minimum: e.g. strategy c) needs slightly less because it does not always
    fill all the slots (A_c < 1).

    Inputs:
        n_slots         : satellite positions in the plane [-]
        reliability     : dict with the reliability model
        design_life_s   : design life [s]
        commissioning_s : commissioning time [s]
    Output:
        launch rate [satellites/s]
    """
    def R_of_years(t_years):
        return reliability_analytic(t_years * YEAR_S, **reliability)

    e_up_s = YEAR_S * quad(R_of_years, commissioning_s / YEAR_S, design_life_s / YEAR_S,
                           limit=200)[0]
    return n_slots / e_up_s
