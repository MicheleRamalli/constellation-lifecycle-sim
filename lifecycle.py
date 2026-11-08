"""Lifecycle Monte Carlo of one orbital plane, strategy "launch on failure".

Model (all times in seconds):
- The plane has n_slots positions. At t = 0 a satellite is launched in every slot.
- At launch, each satellite gets a failure time drawn from the reliability model.
  Age is counted from launch (Castet & Saleh measure R(t) from launch), so a
  satellite can also fail during commissioning.
- A satellite is operational from launch + commissioning until its end of life,
  which is the earliest of: on-orbit failure, retirement at design life.
  A failed launch ends the satellite at the launch time.
- When the end of a satellite is detected, a replacement is ordered; it is
  launched order_to_launch later, into the same slot.
- Fixed time step dt. Events are detected at the first time step at or after
  they happen (detection delay between 0 and dt).

Each Monte Carlo run has its own random generator, spawned from one master seed
with np.random.SeedSequence, so every run is reproducible and independent.
"""

import numpy as np

from reliability import sample_failure_times

# Codes for how a satellite ended its life
ON_ORBIT_FAILURE = 0
RETIREMENT = 1
LAUNCH_FAILURE = 2


def launch_satellites(n, t_launch_s, rng, design_life_s, launch_success_prob, reliability):
    """Launch n satellites at the same time and draw their end of life.

    Inputs:
        n                   : number of satellites launched [-]
        t_launch_s          : launch time [s]
        rng                 : numpy Generator of this Monte Carlo run
        design_life_s       : design life [s] (np.inf = no retirement)
        launch_success_prob : probability that a launch succeeds [-]
        reliability         : dict with the reliability model, e.g.
                              {"model": "weibull", "beta": ..., "theta_s": ...}
    Outputs:
        t_end_s  : end-of-life time of each satellite [s], shape (n,)
        end_kind : ON_ORBIT_FAILURE, RETIREMENT or LAUNCH_FAILURE, shape (n,)
    """
    success = rng.random(n) < launch_success_prob
    t_fail_s = sample_failure_times(n, rng=rng, **reliability)  # age at failure [s]
    life_s = np.minimum(t_fail_s, design_life_s)
    t_end_s = t_launch_s + np.where(success, life_s, 0.0)
    end_kind = np.where(t_fail_s < design_life_s, ON_ORBIT_FAILURE, RETIREMENT)
    end_kind = np.where(success, end_kind, LAUNCH_FAILURE)
    return t_end_s, end_kind


def simulate_launch_on_failure(n_slots, n_runs, seed, sim_duration_s, dt_s,
                               order_to_launch_s, commissioning_s, design_life_s,
                               launch_success_prob, sats_per_launch, reliability):
    """Monte Carlo simulation of one plane with the launch-on-failure strategy.

    Inputs:
        n_slots             : number of satellite positions in the plane [-]
        n_runs              : number of Monte Carlo runs [-]
        seed                : master seed [-]
        sim_duration_s      : simulated time span [s]
        dt_s                : time step [s]
        order_to_launch_s   : delay from replacement order to launch [s]
        commissioning_s     : delay from launch to operational [s]
        design_life_s       : design life [s] (np.inf = no retirement)
        launch_success_prob : probability that a launch succeeds [-]
        sats_per_launch     : satellites per launch [-] (only 1 is supported)
        reliability         : dict with the reliability model (see launch_satellites)
    Output: dict with
        "t_s"               : time grid [s], shape (n_steps,)
        "n_active"          : operational satellites, shape (n_runs, n_steps)
        "n_launched"        : satellites launched at each step, shape (n_runs, n_steps)
        "n_failures"        : on-orbit failures per run, shape (n_runs,)
        "n_retirements"     : retirements at design life per run, shape (n_runs,)
        "n_launch_failures" : failed launches per run, shape (n_runs,)
    """
    if sats_per_launch != 1:
        raise ValueError("Launch on failure: only one satellite per launch is supported")

    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_runs)]
    n_steps = int(sim_duration_s // dt_s) + 1  # grid includes t = 0
    t_grid_s = np.arange(n_steps) * dt_s

    # State of every slot, shape (n_runs, n_slots)
    shape = (n_runs, n_slots)
    waiting = np.ones(shape, dtype=bool)       # slot empty, replacement ordered
    t_next_launch_s = np.zeros(shape)          # launch time of the ordered replacement
    t_launch_s = np.zeros(shape)               # launch time of the current satellite
    t_end_s = np.zeros(shape)                  # end-of-life time of the current satellite
    end_kind = np.zeros(shape, dtype=int)      # how the current satellite will end

    # Outputs
    n_active = np.zeros((n_runs, n_steps), dtype=np.int16)
    n_launched = np.zeros((n_runs, n_steps), dtype=np.int16)
    n_failures = np.zeros(n_runs, dtype=int)
    n_retirements = np.zeros(n_runs, dtype=int)
    n_launch_failures = np.zeros(n_runs, dtype=int)

    for k in range(n_steps):
        t = t_grid_s[k]

        # 1. Detect satellites that reached their end of life, order replacements
        ended = ~waiting & (t >= t_end_s)
        n_failures += np.sum(ended & (end_kind == ON_ORBIT_FAILURE), axis=1)
        n_retirements += np.sum(ended & (end_kind == RETIREMENT), axis=1)
        n_launch_failures += np.sum(ended & (end_kind == LAUNCH_FAILURE), axis=1)
        waiting[ended] = True
        t_next_launch_s[ended] = t + order_to_launch_s

        # 2. Launch the replacements that are due (each run uses its own generator)
        due = waiting & (t >= t_next_launch_s)
        for r in np.nonzero(due.any(axis=1))[0]:
            slots = np.nonzero(due[r])[0]
            t_end_s[r, slots], end_kind[r, slots] = launch_satellites(
                len(slots), t, rngs[r], design_life_s, launch_success_prob, reliability)
        t_launch_s[due] = t
        waiting[due] = False
        n_launched[:, k] = np.sum(due, axis=1)

        # 3. Count operational satellites: commissioned and not yet ended
        active = ~waiting & (t >= t_launch_s + commissioning_s) & (t < t_end_s)
        n_active[:, k] = np.sum(active, axis=1)

    return {
        "t_s": t_grid_s,
        "n_active": n_active,
        "n_launched": n_launched,
        "n_failures": n_failures,
        "n_retirements": n_retirements,
        "n_launch_failures": n_launch_failures,
    }


def plane_availability(n_active, min_active):
    """Fraction of time steps with at least min_active operational satellites.

    Inputs:
        n_active   : operational satellites, shape (n_runs, n_steps)
        min_active : minimum number of operational satellites required [-]
    Output:
        availability of each run [-], shape (n_runs,)
    """
    return np.mean(n_active >= min_active, axis=1)


# Status codes of a unit of the pool (strategy b)
DEAD = 0           # satellite lost, replacement not yet ordered
WAITING = 1        # replacement ordered, waiting for launch
COMMISSIONING = 2  # launched, not yet commissioned
ACTIVE = 3         # operational in a slot
SPARE = 4          # commissioned, dormant in orbit


def simulate_in_orbit_spares(n_slots, n_spares, reorder_level, n_runs, seed, sim_duration_s,
                             dt_s, order_to_launch_s, commissioning_s, design_life_s,
                             launch_success_prob, sats_per_launch, reliability):
    """Monte Carlo simulation of one plane with in-orbit spares, (s, S) reorder policy.

    Policy: S = n_spares (order-up-to level), s = reorder_level (reorder point).
    Inventory position = spares in orbit + satellites ordered or in commissioning
    - empty slots. When the position is STRICTLY below reorder_level ("<"),
    n_spares - position satellites are ordered and launched together after
    order_to_launch. With n_spares = reorder_level = 0 this is launch on failure.

    Pool representation: the plane is a pool of n_slots + n_spares units. Every
    unit is a satellite (active, spare, in commissioning) or a lost satellite
    (dead, waiting for its replacement). Since orders always bring the position
    back to S, the number of dead + waiting units never exceeds n_slots + n_spares,
    and the position equals n_spares - (number of DEAD units).

    Hypotheses (ASSUMPTION, see params.py):
    H1 a dormant spare fails with the same R(t) as an active satellite, age from launch;
    H2 design life counted from launch, an unused spare is retired at design life;
    H3 the oldest spare takes over first (FIFO);
    H4 takeover is instantaneous (no commissioning, the spare is already in orbit).
    A launched satellite must complete commissioning before becoming a spare or
    filling a slot.

    Inputs: as simulate_launch_on_failure, plus
        n_spares      : order-up-to level S, max spares in orbit [-]
        reorder_level : reorder point s [-], 0 <= s <= S
    Output: dict as simulate_launch_on_failure, plus
        "n_spares_in_orbit" : commissioned dormant spares, shape (n_runs, n_steps)
        "n_spares_lost"     : spares lost while dormant (failure or retirement), shape (n_runs,)
    """
    if sats_per_launch != 1:
        raise ValueError("Only one satellite per launch is supported")
    if not 0 <= reorder_level <= n_spares:
        raise ValueError("The reorder level must satisfy 0 <= reorder_level <= n_spares")

    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_runs)]
    n_steps = int(sim_duration_s // dt_s) + 1
    t_grid_s = np.arange(n_steps) * dt_s

    # State of every unit, shape (n_runs, n_units). At t = 0 all units are
    # waiting for the initial launch.
    shape = (n_runs, n_slots + n_spares)
    status = np.full(shape, WAITING, dtype=np.int8)
    t_next_launch_s = np.zeros(shape)
    t_launch_s = np.zeros(shape)
    t_end_s = np.zeros(shape)
    end_kind = np.zeros(shape, dtype=int)

    n_active = np.zeros((n_runs, n_steps), dtype=np.int16)
    n_spares_in_orbit = np.zeros((n_runs, n_steps), dtype=np.int16)
    n_launched = np.zeros((n_runs, n_steps), dtype=np.int16)
    n_failures = np.zeros(n_runs, dtype=int)
    n_retirements = np.zeros(n_runs, dtype=int)
    n_launch_failures = np.zeros(n_runs, dtype=int)
    n_spares_lost = np.zeros(n_runs, dtype=int)

    for k in range(n_steps):
        t = t_grid_s[k]

        # 1. Detect satellites that reached their end of life (active, spare or
        #    in commissioning)
        alive = (status == COMMISSIONING) | (status == ACTIVE) | (status == SPARE)
        ended = alive & (t >= t_end_s)
        n_failures += np.sum(ended & (end_kind == ON_ORBIT_FAILURE), axis=1)
        n_retirements += np.sum(ended & (end_kind == RETIREMENT), axis=1)
        n_launch_failures += np.sum(ended & (end_kind == LAUNCH_FAILURE), axis=1)
        n_spares_lost += np.sum(ended & (status == SPARE), axis=1)
        status[ended] = DEAD

        # 2. Reorder: position = n_spares - dead units; reorder if position < s.
        #    The order quantity S - position is exactly the number of dead units.
        position = n_spares - np.sum(status == DEAD, axis=1)
        reorder = position < reorder_level
        to_order = (status == DEAD) & reorder[:, None]
        status[to_order] = WAITING
        t_next_launch_s[to_order] = t + order_to_launch_s

        # 3. Launch the satellites that are due (each run uses its own generator)
        due = (status == WAITING) & (t >= t_next_launch_s)
        for r in np.nonzero(due.any(axis=1))[0]:
            units = np.nonzero(due[r])[0]
            t_end_s[r, units], end_kind[r, units] = launch_satellites(
                len(units), t, rngs[r], design_life_s, launch_success_prob, reliability)
        t_launch_s[due] = t
        status[due] = COMMISSIONING
        n_launched[:, k] = np.sum(due, axis=1)

        # 4. Commissioned satellites (still alive) join the spares
        ready = (status == COMMISSIONING) & (t >= t_launch_s + commissioning_s) & (t < t_end_s)
        status[ready] = SPARE

        # 5. Spares fill the empty slots, oldest first (H3), instantly (H4)
        n_empty = n_slots - np.sum(status == ACTIVE, axis=1)
        n_available = np.sum(status == SPARE, axis=1)
        for r in np.nonzero((n_empty > 0) & (n_available > 0))[0]:
            spares = np.nonzero(status[r] == SPARE)[0]
            oldest_first = spares[np.argsort(t_launch_s[r, spares], kind="stable")]
            status[r, oldest_first[:n_empty[r]]] = ACTIVE

        n_active[:, k] = np.sum(status == ACTIVE, axis=1)
        n_spares_in_orbit[:, k] = np.sum(status == SPARE, axis=1)

    return {
        "t_s": t_grid_s,
        "n_active": n_active,
        "n_spares_in_orbit": n_spares_in_orbit,
        "n_launched": n_launched,
        "n_failures": n_failures,
        "n_retirements": n_retirements,
        "n_launch_failures": n_launch_failures,
        "n_spares_lost": n_spares_lost,
    }
