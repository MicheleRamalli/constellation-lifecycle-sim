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

# --- Simulation ---
SIM_DURATION_YEARS = 15.0  # ASSUMPTION: reference mission horizon, to be reviewed
DT_DAYS = 1.0              # fixed time step (project rule, CLAUDE.md)
N_RUNS = 1000              # number of Monte Carlo runs
SEED = 42                  # master seed of the Monte Carlo
