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

# --- Constellation geometry ---
# From the Walker search of coverage.py (smallest constellation meeting the revisit
# requirement with ETA_MAX_DEG and MIN_ACCESS_S): Walker 2/4/0, star, SSO 500 km.
N_PLANES = 2   # planes P (coverage.py Walker search)
N_SLOTS = 4    # satellites per plane S (coverage.py Walker search)
MIN_ACTIVE = N_SLOTS  # per-plane metric only; the constellation availability is defined
                      # on the revisit (coverage.py lookup table)

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
DEPLOY_BATCHES = 3           # ASSUMPTION: as Eaglet II (3 batches); used by the single-plane
                             # analyses and tests of v0.1
DEPLOY_INTERVAL_DAYS = 180.0 # ASSUMPTION: constant interval, rounded from the Eaglet II ~4 and ~6 months
# Constellation deployment (D1): plane p (p = 0, 1, ...) is launched in one launch at
# p * DEPLOY_INTERVAL_DAYS with its N_SLOTS + spares.
# ASSUMPTION: one launch per plane (changing plane costs too much delta-v); interval
# consistent with IRIDE Eaglet II batches.

# --- Replacement strategy: launch on failure ---
ORDER_TO_LAUNCH_DAYS = 365.0  # ASSUMPTION: provisional reference value, time from replacement order to launch (no source yet)
COMMISSIONING_DAYS = 30.0     # ASSUMPTION: provisional reference value, time from launch to operational (no source yet)
# Launch success (v0.3): one draw per launch, a failed launch loses all its satellites.
# Source: J. McDowell, GCAT launch list, https://planet4589.org/space/gcat/tsv/launch/launch.tsv,
# downloaded 2026-11-09 (file "Updated 2026 Nov 9"; last Falcon 9 entry 2026-11-07); counts by
# launch_log.py. Falcon 9 Block 5 = GCAT Variant "FT5" (first entry 2018-05-11, booster B1046,
# the first Block 5). Orbital + deep-space launches only (the suborbital Dragon in-flight
# abort is excluded). The only failure is Starlink Group 9-3 (2024-07-12, code OF40, payload
# re-entered): counted as a failure. Estimator: Jeffreys, posterior mean (k + 1/2)/(n + 1).
LAUNCH_LOG_RAW = "inputs/launch_log/gcat_launch_2026-11-09.tsv"  # raw, NOT in git
LAUNCH_LV_TYPE = "Falcon 9"
LAUNCH_VARIANT = "FT5"
LAUNCH_N = 644            # Falcon 9 Block 5 launches, 2018-05-11 to 2026-11-07
LAUNCH_N_FAIL = 1
LAUNCH_N_TRANSPORTER = 18       # Transporter-1 to -18 (subset, reported only)
LAUNCH_N_FAIL_TRANSPORTER = 0
LAUNCH_SUCCESS_PROB = (LAUNCH_N - LAUNCH_N_FAIL + 0.5) / (LAUNCH_N + 1)  # Jeffreys mean, 0.99767
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
# k satellites every T days per plane (first cadence launch together with the plane's
# deployment launch); extra satellites stay in orbit as hot spares (H1-H4).
# ASSUMPTION: the cadence of plane 2 is shifted by T/2 (spreads production and launches
# over time).
# H5 ASSUMPTION: k satellites are always ready at each cadence date (production
#    planned in advance, no order-to-launch delay).
CADENCE_SATS_OPTIONS = (1, 2, 3)  # k values explored in the search of the minimum cadence

# --- Mission requirement for the cadence search ---
AVAIL_TARGET = 0.95      # ASSUMPTION: illustrative mission requirement (plane availability)
AVAIL_CONFIDENCE = 0.95  # ASSUMPTION: illustrative mission requirement (fraction of runs)
MC_CI_LEVEL = 0.95       # confidence level of the binomial interval of the Monte Carlo fractions
                         # (conventional 95 %, numerical choice): a configuration whose interval
                         # contains AVAIL_CONFIDENCE is not distinguishable from the requirement

# --- Search settings (numerical, not physical) ---
CADENCE_GRID_DAYS_PER_SAT = (90, 590, 20)  # sweep of T/k per plane [days]: start, stop, step
                                           # (per-plane rate ~0.6-4 sats/yr; reference ~0.85)
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
MIN_ACCESS_S = 30.0             # ASSUMPTION: an optical sensor needs time to point and acquire
MIN_ACCESS_SWEEP_S = (0.0, 15.0, 30.0, 45.0, 60.0)  # sensitivity sweep of MIN_ACCESS_S [s]

# --- Revisit requirement and Walker search (coverage.py) ---
REVISIT_MAX_HOURS = 12.0   # ASSUMPTION: illustrative mission requirement (max revisit)
REVISIT_LAT_DEG = 45.0     # ASSUMPTION: illustrative mission requirement (latitude, ~Italy)
REVISIT_MEAN_HOURS = 6.0   # ASSUMPTION: illustrative alternative requirement (mean revisit)
# ASSUMPTION (quasi-stationarity): at each time the revisit of the constellation is the
# 30-day revisit of its current configuration (n_1, n_2) from the lookup table; valid
# because the configuration changes over months while the revisit is hours.
# ASSUMPTION: after a loss the satellites left in a plane are re-spaced evenly at once
# (R1, instantaneous rephasing; a real in-plane drift takes weeks).
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

# --- Scenario 2: IRIDE Eaglet II from public TLEs (tle.py, fetch_tle.py, planes.py) ---
# Objects: CelesTrak SATCAT, names "IRIDE-MS1-EAGLET 2-n". 1st launch 2025-276 (EAGLET 2-1..2-8),
# 2nd launch 2026-067 (EAGLET 2-9..2-16). The 3rd launch (2026-227, Transporter-18, 01/10/2026)
# has only 5 of its 8 Eaglet II identifiable by name in the SATCAT (2-17, 2-20, 2-22, 2-23, 2-24);
# 2-18, 2-19, 2-21 are not identifiable (provisional names), so the scenario uses the 16
# satellites of the first two launches (decision: do not guess). Their 6-digit NORAD numbers
# would also not fit the classic TLE format.
IRIDE_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?NAME=IRIDE-MS1-EAGLET&FORMAT=TLE"
IRIDE_NORAD = (66695, 66696, 66697, 66698, 66699, 66700, 66701, 66702,
               68447, 68449, 68450, 68451, 68452, 68453, 68461, 68462)
IRIDE_TLE_RAW_PATTERN = "inputs/tle/celestrak_iride_eaglet2_{date}.tle"  # raw, NOT in git
IRIDE_ELEMENTS_CSV = "inputs/iride_eaglet2_elements.csv"               # derived, in git
# Launch dates -> days from the first launch: 0, 122, 307.
# 28/11/2025: ASI, "IRIDE continues its development: eight more Eaglet II satellites in orbit", 30/03/2026
# 30/03/2026: OHB Italia, https://www.ohb-italia.it/?p=31139 (03/04/2026); LEOP of the 2nd batch
#             completed in ~30 h (this is NOT the commissioning time)
# 01/10/2026: ESA, "Italy's IRIDE completes Eaglet II constellation", 02/10/2026
IRIDE_LAUNCH_DAYS = (0.0, 122.0, 307.0)
PLANE_GROUP_THRESHOLD_DEG = 5.0         # ASSUMPTION: max RAAN gap within a plane (see planes.py)
PLANE_GROUP_SWEEP_DEG = (2.0, 5.0, 10.0)  # sensitivity of the plane grouping

# --- Space debris mitigation requirements (v0.3) ---
# Source: ESA Space Debris Mitigation Requirements, ESSB-ST-U-007, Issue 1 Rev 0, 30/10/2023,
# https://technology.esa.int/upload/media/DGHKMZ_6542582e18e33.pdf (requirement ID in each comment).
# The text read does not cite ISO 27852; for the methods it refers to ESSB-HB-U-002 and
# ECSS-E-ST-10-04.
LEO_LIFETIME_LIMIT_YEARS = 5.0  # 5.4.2.3a.1: orbital lifetime in LEO after end of mission < 5 yr.
                                # Count from the end of life of each object (retirement for
                                # the disposed ones, failure for the failed ones): TO VERIFY on ESSB
CUM_COLLISION_PROB_LIMIT = 1e-3  # 5.4.2.3a.2: cumulative collision probability with objects
                                 # > 1 cm, from end of life to re-entry (MASTER flux x area x time)
DISPOSAL_PROB_REF = 0.9          # 5.4.1.1a: probability of successful disposal > 0.9 (includes
                                 # reliability and impacts); reference value of P_d
DISPOSAL_PROB_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # sensitivity sweep of P_d
CASUALTY_RISK_LIMIT = 1e-4       # 5.5b: on-ground casualty risk of a re-entry < 1e-4 (SARA)
# Large constellations: applicability TO VERIFY (definition of large constellation)
DISPOSAL_PROB_LARGE_CONST = 0.95        # 5.4.1.1b: probability of successful disposal > 0.95
CASUALTY_RISK_LIMIT_LARGE_CONST = 1e-6  # 5.5d.1: casualty risk < 1e-6
APOGEE_LIMIT_LARGE_CONST_KM = 375.0     # 5.4.2.3c: apogee < 375 km (wording TO VERIFY)

# --- Eaglet II satellite (decay.py) ---
# Mass: ESA, 31/03/2026 (~25 kg), https://www.esa.int/Applications/Observing_the_Earth/IRIDE/
# Eight_more_satellites_added_to_IRIDE_space_programme (to be checked that this is the 31/03/2026 article)
EAGLET_MASS_KG = 25.0
EAGLET_BOX_M = (0.5, 0.4, 0.3)  # ASSUMPTION: box "the size of a microwave oven" (ESA); no published
                                # dimensions. Mean area of a tumbling convex body = surface / 4
                                # (Cauchy), solar panels unknown -> sweep below
EAGLET_AREA_FACTORS = (0.5, 1.0, 2.0)  # sensitivity sweep of the area [-]
EAGLET_PROPULSION = True  # ASSUMPTION: no public source (checked ESA, ASI, OHB Italia). Main
                          # argument: free-decay lifetime from 500 km ~2.3 yr (moderate activity,
                          # decay.py) < 5 yr design life (DESIGN_LIFE_YEARS, itself an ASSUMPTION
                          # without source). Weak hint: EAGLET 2-11 has da/dt > 0 (ndot < 0) in
                          # the TLE of 2026-11-09, possibly a fit artefact. A TLE time series
                          # (Space-Track) would be needed to confirm: future work.
# Model (debris.py): operational satellite at constant altitude (ASSUMPTION: station-keeping
# propellant not modelled, relevant mainly at 400-450 km). On-orbit failure = propulsion lost
# -> natural decay from the operational altitude. At retirement a disposal manoeuvre is tried,
# success probability DISPOSAL_PROB_REF (sweep DISPOSAL_PROB_SWEEP), applied ONLY to the
# satellites reaching retirement; the effective P_d = disposed / all ends in orbit is a result,
# compared with ESSB 5.4.1.1a (which includes the system reliability). Failed manoeuvre ->
# natural decay from the operational altitude. Successful manoeuvre -> circular orbit at
# DISPOSAL_ALT_KM, then natural decay. Launch failures leave no object in orbit.
DISPOSAL_ALT_KM = 375.0  # ESSB-ST-U-007 5.4.2.3c (apogee < 375 km, large constellations;
                         # applicability to Eaglet II TO VERIFY). ASSUMPTION: instantaneous
                         # manoeuvre, delta-v and propellant not modelled
DRAG_COEFFICIENT = 2.2     # ASSUMPTION: common value for a compact satellite in free-molecular
                           # flow; same value as the DRAMA/OSCAR Sentinel-3B reference run

# --- MASTER-8 debris flux (debris.py) ---
# Settings and source in the header of the file (MASTER 8.2.3, population data_2602_all,
# SSO i = 97.4 deg, sphere target, debris only, 200-800 km step 50 km, asymmetric 1-sigma).
MASTER_FLUX_CSV = "inputs/master8_flux_sso.csv"
FLUX_SIZE = "1cm"  # ESSB 5.4.2.3a.2: collision probability with objects > 1 cm
# ASSUMPTION: log-linear interpolation of the flux between the 50 km nodes (flux spans orders
# of magnitude). Uncertainty: three separate runs low / nominal / high = nominal - sigma_minus,
# nominal, nominal + sigma_plus, fully correlated over altitude (no symmetric Gaussian).
# ASSUMPTION: flux neglected below 200 km (lowest MASTER node; the object re-enters in days).
# Collision cross-section = mean area of the tumbling box (surface / 4), consistent with the
# sphere target of MASTER; the debris size (>= 1 cm) is neglected against the satellite size.

ALT_TRADE_KM = (400.0, 500.0, 600.0, 700.0, 800.0)  # altitude trade-off (v0.3): range of the plan,
                                                  # inside the MASTER table; 100 km step for run time
DEBRIS_SEED_STREAM = 7  # second entry of the seed sequence [SEED, 7] of the disposal draws
                        # (separate stream: the lifecycle runs are unchanged)

# --- Atmosphere for the drag decay (decay.py) ---
# Piecewise exponential density (Vallado method) through the tabulated densities of
# ECSS-E-ST-10-04C Rev.1 (15 June 2020), Annex G: NRLMSISE-00 Tables G-1 / G-2 / G-3 (low,
# moderate, high long-term solar and geomagnetic activity, pages 144-146) and JB-2006
# Table G-4 (pages 147-148); activity indices in Table 6-3 (page 45). Values identical to the
# 2008 issue. Transcribed in inputs/ecss_density.csv (checked against the PDF: 46 rows, 0
# differences). Constant activity in time: ASSUMPTION (OSCAR uses a time-varying forecast).
# Note: Table G-3 at 900 km gives 6.00e-14 kg/m^3, but the sum of its own constituent
# densities gives 5.23e-14 (all other rows agree within 2.5 %): probably a typo in the
# standard; it only affects 880-900 km, high activity (not used here: orbits <= 800 km).
ATM_DENSITY_CSV = "inputs/ecss_density.csv"
ATM_MODEL = "nrlmsise"            # reference model: NRLMSISE-00, as OSCAR's default atmosphere
ATM_MODEL_SENSITIVITY = "jb2006"  # second model of the same standard, for sensitivity
SOLAR_ACTIVITY_LEVELS = ("low", "moderate", "high")
DECAY_END_ALT_KM = 150.0  # ASSUMPTION: altitude where the orbit is considered re-entered. The
                          # lifetime changes by < 0.01 yr between 100 and 200 km (decay_study in
                          # main.py); OSCAR's value is not stated in the DRAMA 4.1.4 user documentation
DECAY_END_SWEEP_KM = (100.0, 150.0, 200.0)  # sensitivity of the lifetime to the end altitude

# --- Reference case for decay.py: DRAMA 4.1.4 OSCAR, Sentinel-3B tutorial (run by the user) ---
# Circular orbit, C_D 2.2, NRLMSISE-00, ESA "latest prediction" solar activity, start 2018-04-26;
# result 14.28 yr INCLUDING a 5 % lifetime margin (OSCAR documentation: recommended by
# ISO 27852:2024), re-entry 2032-08-08.
S3B_ALT_KM = 600.27
S3B_INC_DEG = 98.62
S3B_AREA_M2 = 16.435
S3B_MASS_KG = 1130.0
S3B_LIFETIME_YEARS = 14.28
S3B_MARGIN = 0.05

# --- Simulation ---
SIM_DURATION_YEARS = 15.0  # ASSUMPTION: reference mission horizon, to be reviewed
DT_DAYS = 1.0              # fixed time step (project rule, CLAUDE.md)
N_RUNS = 1000              # number of Monte Carlo runs
N_RUNS_SWEEP = 1000        # Monte Carlo runs per point of the parameter sweeps in main.py
SEED = 42                  # master seed of the Monte Carlo
