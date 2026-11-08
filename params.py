"""Model parameters. Every value has its source or is marked as ASSUMPTION.

Times are stored here in convenient units (days, years) and converted
to SI (seconds) with SECONDS_PER_DAY / SECONDS_PER_YEAR where they enter the code.
"""

# --- Unit conversion ---
SECONDS_PER_DAY = 86400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY  # Julian year [s], IAU definition

# --- Satellite reliability: Weibull model ---
# Maximum likelihood (MLE) fit on 1584 Earth-orbiting satellites launched
# between January 1990 and October 2008 (dataset described on p. 2).
# Source: Castet J.-F., Saleh J.H. (2009), "Satellite and satellite subsystems
# reliability: Statistical data analysis and modeling",
# Reliability Engineering and System Safety 94, 1718-1728.
# R(t) is measured from the launch date.
WEIBULL_BETA = 0.4521          # shape parameter [-] (beta < 1: infant mortality)
WEIBULL_THETA_YEARS = 2607.0   # scale parameter [years]

# --- Mission ---
DESIGN_LIFE_YEARS = 5.0  # ASSUMPTION: reference value, to be reviewed (no source yet)

# --- Constellation geometry (one orbital plane for v0.1) ---
N_SLOTS = 6  # ASSUMPTION: provisional, will come from the Walker constellation design
MIN_ACTIVE = N_SLOTS  # ASSUMPTION: provisional, full plane required (coverage requirement not yet defined)

# --- Initial deployment in batches ---
# ASSUMPTION inspired by IRIDE Eaglet II: 24 satellites deployed in 3 batches of 8,
# first batch launched on 28 November 2025, second batch reported in orbit in
# March 2026, constellation completed in autumn 2026 (real intervals ~4 and ~6 months).
# The exact dates of the 2nd and 3rd launches are not given in these releases.
# Sources:
# - ASI, "IRIDE continues its development: eight more Eaglet II satellites in orbit",
#   30/03/2026, https://www.asi.it/en/2026/03/iride-continues-its-development-eight-more-eaglet-ii-satellites-in-orbit/
# - ESA, "Eight more satellites added to IRIDE space programme", 31/03/2026,
#   https://www.esa.int/Applications/Observing_the_Earth/IRIDE/Eight_more_satellites_added_to_IRIDE_space_programme
# - ESA, "Italy's IRIDE completes Eaglet II constellation", 02/10/2026,
#   https://www.esa.int/Applications/Observing_the_Earth/IRIDE/Italy_s_IRIDE_completes_Eaglet_II_constellation
# The N_SLOTS + N_SPARES units are split as evenly as possible, larger batches first.
DEPLOY_BATCHES = 3           # ASSUMPTION: as Eaglet II (3 batches)
DEPLOY_INTERVAL_DAYS = 180.0 # ASSUMPTION: constant interval, rounded from the Eaglet II ~4 and ~6 months

# --- Replacement strategy: launch on failure ---
ORDER_TO_LAUNCH_DAYS = 365.0  # ASSUMPTION: provisional reference value, time from replacement order to launch (no source yet)
COMMISSIONING_DAYS = 30.0     # ASSUMPTION: provisional reference value, time from launch to operational (no source yet)
LAUNCH_SUCCESS_PROB = 1.0     # ASSUMPTION: launch failures neglected in v0.1
SATS_PER_LAUNCH = 1           # ASSUMPTION: launch on failure, one dedicated launch per replacement

# --- Replacement strategy b): in-orbit spares in the same plane ---
# (s, S) policy with backorders: reorder when the inventory position
# (spares in orbit + ordered - empty slots) is STRICTLY below REORDER_LEVEL,
# ordering up to N_SPARES.
N_SPARES = 1               # ASSUMPTION: provisional, to be traded off later
REORDER_LEVEL = N_SPARES   # ASSUMPTION: provisional, one reorder per lost satellite (one-for-one)
# Spare modelling hypotheses (documented in lifecycle.simulate_in_orbit_spares):
# H1 ASSUMPTION: a dormant spare has the same Weibull R(t) as an active satellite, age from
#    launch (hot standby). Castet & Saleh measure R(t) from launch for satellites in orbit and
#    no source is available for a reduced dormant failure rate.
# H2 ASSUMPTION: design life is counted from launch for spares too (ageing and station-keeping
#    propellant); a spare reaching its design life unused is retired.
# H3 ASSUMPTION: the oldest available spare takes over first (FIFO).
# H4 ASSUMPTION: takeover is instantaneous (phasing drift within the plane neglected).

# --- Replacement strategy c): early ordering of the retirement replacement ---
# Every live satellite (also dormant spares) orders its successor at age
# DESIGN_LIFE - ORDER_TO_LAUNCH - COMMISSIONING, so the successor enters service
# when it retires. No new parameter: it uses the values above.
# ASSUMPTION: end of life known in advance; mission extensions beyond design life
# are not modelled.

# --- Replacement strategy d): launches at a fixed cadence ---
# k satellites every T days after the initial deployment (first cadence launch at
# last deployment batch + T); extra satellites stay in orbit as hot spares (H1-H4).
# H5 ASSUMPTION: k satellites are always ready at each cadence date (production
#    planned in advance, no order-to-launch delay).
CADENCE_SATS_OPTIONS = (1, 2, 3)  # k values explored in the search of the minimum cadence

# --- Mission requirement for the cadence search ---
AVAIL_TARGET = 0.95      # ASSUMPTION: illustrative mission requirement (plane availability)
AVAIL_CONFIDENCE = 0.95  # ASSUMPTION: illustrative mission requirement (fraction of runs)

# --- Search settings (numerical, not physical) ---
CADENCE_GRID_DAYS_PER_SAT = (90, 365, 15)  # sweep of T/k [days]: start, stop, step (rate ~1-4 sats/yr)
SEEDS_CHECK = (43, 44)                     # extra seeds to check the sensitivity of T*

# --- Earth constants (coverage.py) ---
MU_EARTH = 3.986004418e14   # gravitational parameter [m^3/s^2], WGS84 (to be checked on the reference)
R_EARTH = 6378137.0         # equatorial radius [m], WGS84 (to be checked on the reference)
OMEGA_EARTH = 7.292115e-5   # Earth rotation rate [rad/s], WGS84 (to be checked on the reference)
J2_EARTH = 1.08263e-3       # second zonal harmonic [-], Vallado, Fundamentals of Astrodynamics
                            # and Applications, 4th ed. 2013 (to be checked on the reference)
TROPICAL_YEAR_DAYS = 365.2422  # tropical year [days], sets the sun-synchronous RAAN rate
                               # (to be checked on the reference)

# --- Reference Walker constellation (coverage.py) ---
# Circular sun-synchronous orbit, inclination from the J2 condition (coverage.sso_inclination).
# Source for the altitude: OHB Italia, "OHB Italia takes part to the IRIDE satellite
# constellation to protect the Earth" (Eaglet II nominal altitude around 500 km),
# https://www.ohb-italia.it/?p=8404 ; consistent with ESA 31/03/2026 (467-625 km).
ALT_KM = 500.0
ECCENTRICITY = 0.0       # ASSUMPTION: circular orbits
RAAN_SPREAD_DEG = 180.0  # ASSUMPTION: Walker star (planes over 180 deg of RAAN). In SSO it puts
                         # the descending passes on the day side; a 360 deg delta would put half
                         # of the planes on the night side, meaningless for an optical sensor.
# ASSUMPTION: solar illumination not modelled (geometric coverage only); future development.
# ASSUMPTION: spherical Earth for the coverage geometry.
ETA_MAX_DEG = 30.0              # ASSUMPTION: sensor half-angle (max off-nadir); no source for Eaglet II
ETA_SWEEP_DEG = (15.0, 30.0, 45.0)  # sensitivity sweep of ETA_MAX_DEG [deg]

# --- Revisit requirement and Walker search (coverage.py) ---
REVISIT_MAX_HOURS = 12.0   # ASSUMPTION: illustrative mission requirement (max revisit)
REVISIT_LAT_DEG = 45.0     # ASSUMPTION: illustrative mission requirement (latitude, ~Italy)
WALKER_P_RANGE = (1, 6)    # planes explored in the Walker search (numerical setting)
WALKER_S_RANGE = (2, 8)    # satellites per plane explored (numerical setting)

# --- Coverage numerics (convergence study on Walker 3/3/1 and 3/4/1, eta = 30 deg, 45 deg N) ---
# Max revisit identical (to 1e-4 h) for step 5-10 s, duration 30-90 days, longitude
# step 0.5-2 deg; mean revisit within 0.1 %. A 30 s step misses the short accesses at
# the footprint edge (lambda = 2.6 deg) and overestimates the max revisit (10.65 h vs 8.98 h).
COV_STEP_S = 10.0          # propagation time step [s]
COV_DURATION_DAYS = 30.0   # propagation duration [days]
COV_LAT_GRID_DEG = (0.0, 80.0, 5.0)  # latitude grid: start, stop, step [deg]
COV_LON_STEP_DEG = 2.0     # longitude grid step [deg]

# --- Simulation ---
SIM_DURATION_YEARS = 15.0  # ASSUMPTION: reference mission horizon, to be reviewed
DT_DAYS = 1.0              # fixed time step (project rule, CLAUDE.md)
N_RUNS = 1000              # number of Monte Carlo runs
N_RUNS_SWEEP = 1000        # Monte Carlo runs per point of the parameter sweeps in main.py
SEED = 42                  # master seed of the Monte Carlo
