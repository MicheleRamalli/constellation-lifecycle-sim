"""Lifecycle Monte Carlo of one orbital plane.

Three simulators:
- simulate_launch_on_failure: strategy a) only, one satellite per slot. Kept as
  a simple, independent reference implementation.
- simulate_plane: strategies a), b) in-orbit spares, c) early ordering of the
  retirement replacement, and b + c, with one pool of satellite records.
- simulate_fixed_cadence: strategy d), launches at a fixed cadence.
The last two share the helper functions detect_ends, launch_due and
commission_and_fill_slots.

Common model (all times in seconds):
- Each satellite gets its failure time at launch. Age is counted from launch
  (Castet & Saleh measure R(t) from launch), so a satellite can also fail during
  commissioning. A failed launch ends the satellite at the launch time.
- A satellite is operational from launch + commissioning until its end of life,
  the earliest of on-orbit failure and retirement at design life.
- Fixed time step dt. Events are detected at the first time step at or after
  they happen (detection delay between 0 and dt). Launches happen on the grid.
- Initial deployment: the first launch of each unit is given by initial_launch_s
  (e.g. from deployment_launch_times for a deployment in batches).

Each Monte Carlo run has its own random generator, spawned from one master seed
with np.random.SeedSequence, so every run is reproducible and independent.
"""

import numpy as np

from reliability import sample_failure_times

# Codes for how a satellite ended its life
ON_ORBIT_FAILURE = 0
RETIREMENT = 1
LAUNCH_FAILURE = 2

# Status codes of a satellite record (simulate_plane)
FREE = 0           # record not in use
DEAD = 1           # satellite lost, replacement not yet ordered
WAITING = 2        # satellite ordered, waiting for launch
COMMISSIONING = 3  # launched, not yet commissioned
ACTIVE = 4         # operational in a slot
SPARE = 5          # commissioned, dormant in orbit


def deployment_launch_times(n_units, n_batches, interval_s):
    """First launch time of each unit for a deployment in equally spaced batches.

    The units are split as evenly as possible, larger batches first
    (e.g. 7 units in 3 batches -> 3, 2, 2). Batch b is launched at b * interval.

    Inputs:
        n_units    : satellites of the initial deployment [-]
        n_batches  : number of launch batches [-]
        interval_s : time between two batches [s]
    Output:
        launch time of each unit [s], shape (n_units,)
    """
    batches = np.array_split(np.arange(n_units), n_batches)
    t_launch_s = np.zeros(n_units)
    for b, units in enumerate(batches):
        t_launch_s[units] = b * interval_s
    return t_launch_s


def cadence_launch_times(cadence_start_s, cadence_interval_s, sim_duration_s):
    """Dates of the fixed-cadence launches within the simulated time span.

    Inputs:
        cadence_start_s    : first cadence launch [s]
        cadence_interval_s : time between cadence launches [s]
        sim_duration_s     : simulated time span [s]
    Output:
        launch dates [s], shape (n_launches,)
    """
    if cadence_start_s > sim_duration_s:
        return np.zeros(0)
    n_launches = int((sim_duration_s - cadence_start_s) // cadence_interval_s) + 1
    return cadence_start_s + np.arange(n_launches) * cadence_interval_s


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


def simulate_launch_on_failure(n_slots, n_runs, seed, sim_duration_s, dt_s, initial_launch_s,
                               order_to_launch_s, commissioning_s, design_life_s,
                               launch_success_prob, sats_per_launch, reliability):
    """Monte Carlo simulation of one plane with the launch-on-failure strategy.

    When the end of a satellite is detected, a replacement is ordered and
    launched order_to_launch later into the same slot.

    Inputs:
        n_slots             : number of satellite positions in the plane [-]
        n_runs              : number of Monte Carlo runs [-]
        seed                : master seed [-]
        sim_duration_s      : simulated time span [s]
        dt_s                : time step [s]
        initial_launch_s    : first launch time of each slot [s], shape (n_slots,)
                              or (n_runs, n_slots)
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
    waiting = np.ones(shape, dtype=bool)       # slot empty, satellite ordered
    t_next_launch_s = np.broadcast_to(initial_launch_s, shape).astype(float)  # (a copy)
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

        # 2. Launch the satellites that are due (each run uses its own generator)
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


def is_alive(status):
    """True for records holding a live satellite (in commissioning, active or spare)."""
    return (status == COMMISSIONING) | (status == ACTIVE) | (status == SPARE)


def new_counters(n_runs):
    """Per-run event counters shared by simulate_plane and simulate_fixed_cadence."""
    keys = ["n_failures", "n_retirements", "n_launch_failures", "n_spares_lost"]
    return {key: np.zeros(n_runs, dtype=int) for key in keys}


def detect_ends(t, status, t_end_s, end_kind, counters):
    """Find the live satellites whose end of life is reached at time t.

    Inputs:
        t                : current time [s]
        status, t_end_s, end_kind : record arrays, shape (n_runs, n_records)
        counters         : dict of per-run counters (new_counters), updated in place
    Output:
        ended : boolean mask, shape (n_runs, n_records); the caller sets the new status
    """
    ended = is_alive(status) & (t >= t_end_s)
    counters["n_failures"] += np.sum(ended & (end_kind == ON_ORBIT_FAILURE), axis=1)
    counters["n_retirements"] += np.sum(ended & (end_kind == RETIREMENT), axis=1)
    counters["n_launch_failures"] += np.sum(ended & (end_kind == LAUNCH_FAILURE), axis=1)
    counters["n_spares_lost"] += np.sum(ended & (status == SPARE), axis=1)
    return ended


def launch_due(t, status, t_next_launch_s, t_launch_s, t_end_s, end_kind, rngs,
               design_life_s, launch_success_prob, reliability):
    """Launch the WAITING records due at time t (each run uses its own generator).

    The record arrays are updated in place: the launched records become
    COMMISSIONING with their launch time and end of life.
    Output: satellites launched in each run, shape (n_runs,)
    """
    due = (status == WAITING) & (t >= t_next_launch_s)
    for r in np.nonzero(due.any(axis=1))[0]:
        records = np.nonzero(due[r])[0]
        t_end_s[r, records], end_kind[r, records] = launch_satellites(
            len(records), t, rngs[r], design_life_s, launch_success_prob, reliability)
    t_launch_s[due] = t
    status[due] = COMMISSIONING
    return np.sum(due, axis=1)


def commission_and_fill_slots(t, status, t_launch_s, t_end_s, commissioning_s, n_slots):
    """Commissioned satellites become spares; spares fill the empty slots.

    Satellites still alive after commissioning join the spares. Then spares fill
    the empty slots, oldest first (H3), instantly (H4). status is updated in place.
    """
    ready = (status == COMMISSIONING) & (t >= t_launch_s + commissioning_s) & (t < t_end_s)
    status[ready] = SPARE
    n_empty = n_slots - np.sum(status == ACTIVE, axis=1)
    n_available = np.sum(status == SPARE, axis=1)
    for r in np.nonzero((n_empty > 0) & (n_available > 0))[0]:
        spares = np.nonzero(status[r] == SPARE)[0]
        oldest_first = spares[np.argsort(t_launch_s[r, spares], kind="stable")]
        status[r, oldest_first[:n_empty[r]]] = ACTIVE


def simulate_plane(n_slots, n_spares, reorder_level, pre_order_retirement, n_runs, seed,
                   sim_duration_s, dt_s, initial_launch_s, order_to_launch_s, commissioning_s,
                   design_life_s, launch_success_prob, sats_per_launch, reliability):
    """Monte Carlo simulation of one plane: spares (s, S) policy and early ordering.

    Strategies:
        a) launch on failure : n_spares = reorder_level = 0, pre_order_retirement = False
        b) in-orbit spares   : n_spares = S, reorder_level = s, pre_order_retirement = False
        c) early ordering    : n_spares = reorder_level = 0, pre_order_retirement = True
        b + c                : n_spares = S, reorder_level = s, pre_order_retirement = True

    Unplanned losses, (s, S) policy with backorders: S = n_spares (order-up-to
    level), s = reorder_level (reorder point). Inventory position = S - (number
    of DEAD records), i.e. spares in orbit + satellites ordered or in
    commissioning - empty slots, counting only losses without a planned
    successor. When the position is STRICTLY below reorder_level ("<"), all DEAD
    records are reordered (quantity S - position) and launched together after
    order_to_launch.

    Early ordering (pre_order_retirement = True): every live satellite (active,
    spare or in commissioning) orders its own successor when its age reaches
    a = design_life - order_to_launch - commissioning. The successor is launched
    at age design_life - commissioning and is commissioned exactly when the old
    satellite retires. A satellite that ends with a successor already ordered
    frees its record without any reorder. A successor lost at launch or during
    commissioning is a normal unplanned loss.

    Records: the pool holds 2 * (n_slots + n_spares) satellite records, enough
    for each unit to hold an old satellite and its successor at the same time.
    The units initially deployed are the first n_slots + n_spares records.

    Spare hypotheses (ASSUMPTION, see params.py):
    H1 a dormant spare fails with the same R(t) as an active satellite, age from launch;
    H2 design life counted from launch, an unused spare is retired at design life;
    H3 the oldest spare takes over first (FIFO);
    H4 takeover is instantaneous (no commissioning, the spare is already in orbit).
    A launched satellite must complete commissioning before becoming a spare or
    filling a slot.

    Inputs: as simulate_launch_on_failure, plus
        n_spares             : order-up-to level S [-]
        reorder_level        : reorder point s [-], 0 <= s <= S
        pre_order_retirement : True to order the retirement replacement in advance
        initial_launch_s     : first launch time of each unit [s], shape
                               (n_slots + n_spares,) or (n_runs, n_slots + n_spares)
    Output: dict as simulate_launch_on_failure, plus
        "n_spares_in_orbit" : commissioned dormant spares, shape (n_runs, n_steps)
        "n_spares_lost"     : spares lost while dormant (failure or retirement), shape (n_runs,)
    """
    if sats_per_launch != 1:
        raise ValueError("Only one satellite per launch is supported")
    if not 0 <= reorder_level <= n_spares:
        raise ValueError("The reorder level must satisfy 0 <= reorder_level <= n_spares")
    pre_order_age_s = design_life_s - order_to_launch_s - commissioning_s
    if pre_order_retirement and pre_order_age_s < commissioning_s:
        # With a >= commissioning, a successor never orders its own successor
        # before its predecessor has retired, so two records per unit are enough.
        raise ValueError("Early ordering needs design life >= order_to_launch + 2 commissioning")

    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_runs)]
    n_steps = int(sim_duration_s // dt_s) + 1
    t_grid_s = np.arange(n_steps) * dt_s

    n_units = n_slots + n_spares
    shape = (n_runs, 2 * n_units)
    status = np.full(shape, FREE, dtype=np.int8)
    status[:, :n_units] = WAITING
    t_next_launch_s = np.zeros(shape)
    t_next_launch_s[:, :n_units] = np.broadcast_to(initial_launch_s, (n_runs, n_units))
    t_launch_s = np.zeros(shape)
    t_end_s = np.zeros(shape)
    end_kind = np.zeros(shape, dtype=int)
    has_successor = np.zeros(shape, dtype=bool)

    n_active = np.zeros((n_runs, n_steps), dtype=np.int16)
    n_spares_in_orbit = np.zeros((n_runs, n_steps), dtype=np.int16)
    n_launched = np.zeros((n_runs, n_steps), dtype=np.int16)
    counters = new_counters(n_runs)

    for k in range(n_steps):
        t = t_grid_s[k]

        # 1. Detect satellites that reached their end of life
        ended = detect_ends(t, status, t_end_s, end_kind, counters)
        status[ended & has_successor] = FREE   # planned: the successor is on its way
        status[ended & ~has_successor] = DEAD  # unplanned loss
        has_successor[ended] = False

        # 2. Unplanned losses: (s, S) reorder if position < s
        position = n_spares - np.sum(status == DEAD, axis=1)
        reorder = position < reorder_level
        to_order = (status == DEAD) & reorder[:, None]
        status[to_order] = WAITING
        t_next_launch_s[to_order] = t + order_to_launch_s

        # 3. Early ordering: live satellites at age >= a order their successor
        if pre_order_retirement:
            planned = is_alive(status) & ~has_successor & (t >= t_launch_s + pre_order_age_s)
            for r in np.nonzero(planned.any(axis=1))[0]:
                n_new = np.sum(planned[r])
                free = np.nonzero(status[r] == FREE)[0]
                if len(free) < n_new:
                    raise RuntimeError("No free satellite record for a successor")
                status[r, free[:n_new]] = WAITING
                t_next_launch_s[r, free[:n_new]] = t + order_to_launch_s
            has_successor[planned] = True

        # 4. Launch the satellites that are due
        n_launched[:, k] = launch_due(t, status, t_next_launch_s, t_launch_s, t_end_s, end_kind,
                                      rngs, design_life_s, launch_success_prob, reliability)

        # 5-6. Commissioning, spares fill the empty slots
        commission_and_fill_slots(t, status, t_launch_s, t_end_s, commissioning_s, n_slots)

        n_active[:, k] = np.sum(status == ACTIVE, axis=1)
        n_spares_in_orbit[:, k] = np.sum(status == SPARE, axis=1)

    return {"t_s": t_grid_s, "n_active": n_active, "n_spares_in_orbit": n_spares_in_orbit,
            "n_launched": n_launched, **counters}


def simulate_fixed_cadence(n_slots, cadence_sats, cadence_interval_s, cadence_start_s, n_runs,
                           seed, sim_duration_s, dt_s, initial_launch_s, commissioning_s,
                           design_life_s, launch_success_prob, reliability):
    """Monte Carlo simulation of one plane with launches at a fixed cadence (strategy d).

    After the initial deployment (initial_launch_s), cadence_sats satellites are
    launched every cadence_interval, starting at cadence_start, whatever happens
    in orbit. Lost satellites are never reordered. Satellites not needed in a
    slot stay in orbit as hot spares (H1-H4, see simulate_plane) and retire at
    design life from their own launch, even if never used.
    H5 ASSUMPTION: cadence_sats satellites are always ready at each cadence date
    (production planned in advance, no order-to-launch delay).

    Records: a satellite lives at most design_life, so at any time at most
    n_initial + cadence_sats * (ceil(DL / interval) + 1) satellites exist; records
    of ended satellites are freed and reused (+ one cohort of margin).

    Inputs:
        n_slots             : satellite positions in the plane [-]
        cadence_sats        : satellites per cadence launch, k [-]
        cadence_interval_s  : time between cadence launches, T [s]
        cadence_start_s     : time of the first cadence launch [s]
        n_runs, seed, sim_duration_s, dt_s : as simulate_plane
        initial_launch_s    : first launch time of each initially deployed satellite [s],
                              shape (n_initial,) or (n_runs, n_initial)
        commissioning_s, design_life_s, launch_success_prob, reliability : as simulate_plane
    Output: dict as simulate_plane
    """
    if not np.isfinite(design_life_s):
        raise ValueError("Fixed cadence needs a finite design life (bounded number of records)")

    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n_runs)]
    n_steps = int(sim_duration_s // dt_s) + 1
    t_grid_s = np.arange(n_steps) * dt_s

    n_initial = np.shape(initial_launch_s)[-1]
    n_records = n_initial + cadence_sats * (int(np.ceil(design_life_s / cadence_interval_s)) + 2)
    shape = (n_runs, n_records)
    status = np.full(shape, FREE, dtype=np.int8)
    status[:, :n_initial] = WAITING
    t_next_launch_s = np.zeros(shape)
    t_next_launch_s[:, :n_initial] = np.broadcast_to(initial_launch_s, (n_runs, n_initial))
    t_launch_s = np.zeros(shape)
    t_end_s = np.zeros(shape)
    end_kind = np.zeros(shape, dtype=int)

    n_active = np.zeros((n_runs, n_steps), dtype=np.int16)
    n_spares_in_orbit = np.zeros((n_runs, n_steps), dtype=np.int16)
    n_launched = np.zeros((n_runs, n_steps), dtype=np.int16)
    counters = new_counters(n_runs)
    t_next_cadence_s = cadence_start_s

    for k in range(n_steps):
        t = t_grid_s[k]

        # 1. Detect satellites that reached their end of life: no reorder
        ended = detect_ends(t, status, t_end_s, end_kind, counters)
        status[ended] = FREE

        # 2. Cadence launch (same dates in every run)
        if t >= t_next_cadence_s:
            for r in range(n_runs):
                free = np.nonzero(status[r] == FREE)[0]
                if len(free) < cadence_sats:
                    raise RuntimeError("No free satellite record for a cadence launch")
                status[r, free[:cadence_sats]] = WAITING
                t_next_launch_s[r, free[:cadence_sats]] = t
            t_next_cadence_s += cadence_interval_s

        # 3. Launch the satellites that are due
        n_launched[:, k] = launch_due(t, status, t_next_launch_s, t_launch_s, t_end_s, end_kind,
                                      rngs, design_life_s, launch_success_prob, reliability)

        # 4. Commissioning, spares fill the empty slots
        commission_and_fill_slots(t, status, t_launch_s, t_end_s, commissioning_s, n_slots)

        n_active[:, k] = np.sum(status == ACTIVE, axis=1)
        n_spares_in_orbit[:, k] = np.sum(status == SPARE, axis=1)

    return {"t_s": t_grid_s, "n_active": n_active, "n_spares_in_orbit": n_spares_in_orbit,
            "n_launched": n_launched, **counters}


def plane_availability(n_active, min_active, t_s, t_start_s):
    """Fraction of time steps with at least min_active operational satellites.

    Inputs:
        n_active   : operational satellites, shape (n_runs, n_steps)
        min_active : minimum number of operational satellites required [-]
        t_s        : time grid [s], shape (n_steps,)
        t_start_s  : start of the averaging window [s] (e.g. end of deployment)
    Output:
        availability of each run [-], shape (n_runs,)
    """
    keep = t_s >= t_start_s
    return np.mean(n_active[:, keep] >= min_active, axis=1)


def time_to_full_operation(n_active, min_active, t_s):
    """First time at which n_active >= min_active, for each run.

    Inputs:
        n_active   : operational satellites, shape (n_runs, n_steps)
        min_active : minimum number of operational satellites required [-]
        t_s        : time grid [s], shape (n_steps,)
    Output:
        time [s] of each run, shape (n_runs,); NaN if never reached
    """
    full = n_active >= min_active
    first = np.argmax(full, axis=1)  # index of the first True (0 if none)
    return np.where(full.any(axis=1), t_s[first], np.nan)


# ---------------------------------------------------------------------------
# Constellation of independent planes
# ---------------------------------------------------------------------------

def simulate_constellation(plane_simulator, plane_inputs, seed):
    """Monte Carlo of a constellation of independent planes.

    Every launch serves one plane, and spares, reorders and cadence are per
    plane, so the planes are independent processes: the single-plane simulator
    is called once per plane. Plane p uses the seed sequence [seed, p], so each
    plane has its own reproducible random numbers, and run r of plane 1 is
    independent of run r of plane 2.

    Inputs:
        plane_simulator : simulate_plane or simulate_fixed_cadence
        plane_inputs    : list of dicts, one per plane, with all the inputs of
                          plane_simulator except seed (same n_runs and time grid)
        seed            : master seed [-]
    Output: dict with
        "t_s"        : time grid [s], shape (n_steps,)
        "n_active"   : operational satellites per plane, shape (n_planes, n_runs, n_steps)
        "n_launched" : satellites launched by the constellation, shape (n_runs, n_steps)
        "n_failures", "n_retirements", "n_launch_failures", "n_spares_lost" :
                       constellation totals per run, shape (n_runs,)
    """
    outs = [plane_simulator(seed=[seed, p], **inputs) for p, inputs in enumerate(plane_inputs)]
    result = {"t_s": outs[0]["t_s"],
              "n_active": np.stack([o["n_active"] for o in outs]),
              "n_launched": sum(o["n_launched"].astype(int) for o in outs)}
    for key in ["n_failures", "n_retirements", "n_launch_failures", "n_spares_lost"]:
        result[key] = sum(o[key] for o in outs)
    return result


def constellation_revisit_series(n_active_planes, table_s):
    """Revisit of the constellation at each step from the lookup table.

    revisit(t) = table[n_1(t), ..., n_P(t)] (quasi-stationarity ASSUMPTION, params.py).
    Inputs:
        n_active_planes : operational satellites per plane, shape (n_planes, n_runs, n_steps)
        table_s         : revisit [s] for every (n_1, ..., n_P), shape (S + 1,) * n_planes
    Output:
        revisit [s], shape (n_runs, n_steps)
    """
    return table_s[tuple(n_active_planes)]


def constellation_availability(revisit_s, max_revisit_s, t_s, t_start_s):
    """Fraction of time steps with revisit <= requirement, per run.

    Inputs:
        revisit_s     : revisit of the constellation [s], shape (n_runs, n_steps)
        max_revisit_s : requirement [s]
        t_s           : time grid [s]
        t_start_s     : start of the averaging window [s] (end of deployment)
    Output:
        availability of each run [-], shape (n_runs,)
    """
    keep = t_s >= t_start_s
    return np.mean(revisit_s[:, keep] <= max_revisit_s, axis=1)


def time_to_requirement(revisit_s, max_revisit_s, t_s):
    """First time at which revisit <= requirement, per run [s] (NaN if never)."""
    ok = revisit_s <= max_revisit_s
    first = np.argmax(ok, axis=1)
    return np.where(ok.any(axis=1), t_s[first], np.nan)
