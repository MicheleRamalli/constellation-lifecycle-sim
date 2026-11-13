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
# Sensitivity case "worst reliability" (main.py --reliability): MLE Weibull of the Small
# category (0-500 kg launch mass, section 2, Table 2; 415 of 1444 satellites).
# Source: Dubos G.F., Castet J.-F., Saleh J.H., "Statistical reliability analysis of
# satellites by mass category: Does spacecraft size matter?", IAC-09-D1.3.6, 60th
# International Astronautical Congress, Daejeon, 2009, Table 3, p. 6 (conference version of
# Acta Astronautica 67, 2010: the journal values may differ). Read on 13/11/2026 from the
# public PDF of the Georgia Tech Space Systems Design Lab (ssdl1.gatech.edu,
# IAC-2009-D1.3.6.pdf, no longer online); no local copy. Check: R(6 months) ~ 96.8 % for the
# Small category (p. 5), reproduced in tests/test_reliability.py.
# Limitation: Eaglet II (~25 kg) is far below the 500 kg upper bound of the category.
WEIBULL_BETA_SMALL = 0.3224         # shape parameter [-]
WEIBULL_THETA_SMALL_YEARS = 21414.5  # scale parameter [years]

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
R_EARTH_POLAR = 6356752.3142  # polar radius [m], WGS84 (to be checked on the reference); only
                              # for the oblateness estimate in the OSCAR comparison (main.py)
G0 = 9.80665                # standard gravity [m/s^2] (conventional value, CGPM 1901), rocket equation
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
                                # Counted from the end-of-life epoch for objects with recurrent
                                # manoeuvre capability (from the injection epoch without it):
                                # here retirement for the disposed ones, failure for the failed ones
CUM_COLLISION_PROB_LIMIT = 1e-3  # 5.4.2.3a.2: cumulative collision probability with objects
                                 # > 1 cm, from end of life to re-entry (MASTER flux x area x time)
DISPOSAL_PROB_REF = 0.9          # 5.4.1.1a: probability of successful disposal > 0.9 (includes
                                 # reliability and impacts); reference value of P_d
DISPOSAL_PROB_SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # sensitivity sweep of P_d
CASUALTY_RISK_LIMIT = 1e-4       # 5.5b: expected number of casualties per re-entry < 1e-4 (SARA)
# Large constellations (3.2.25, at least 100 spacecraft; a "constellation" has at least 10, 3.2.8):
# not applicable to IRIDE Eaglet II (24) nor to the reference Walker; values kept for reference.
DISPOSAL_PROB_LARGE_CONST = 0.95        # 5.4.1.1b: probability of successful disposal > 0.95
CASUALTY_RISK_LIMIT_LARGE_CONST = 1e-6  # 5.5d.1: expected casualties per re-entry < 1e-6
                                        # (5.5d.2: or controlled re-entry); per re-entry only,
                                        # ESSB-ST-U-007 gives no cumulative limit
APOGEE_LIMIT_LARGE_CONST_KM = 375.0     # 5.4.2.3c: the disposal orbit of spacecraft of a large
                                        # constellation in the LEO protected region shall have
                                        # an apogee below 375 km
LARGE_CONST_MIN_SATS = 100  # 3.2.25: "group of space objects consisting of at least 100
                            # spacecraft"; IRIDE Eaglet II has 24 satellites

# --- On-ground casualty risk of the re-entries (casualty.py, v0.3) ---
# Casualty area of a fragment A_c = (sqrt(A_h) + sqrt(A_i))^2 with A_h = 0.36 m^2 (sqrt = 0.6 m):
# NASA Orbital Debris Program Office, Debris Assessment Software (DAS) reentry analysis,
# "DCA = (0.6 + sqrt(A))^2" (NASA NTRS 20130011037); requirement NASA-STD-8719.14
# Requirement 4.7-1 (casualty risk of an uncontrolled re-entry <= 1:10000). Note: DAS takes
# A = L*D for cylinders, SARA (and this code) the mean tumbling section surface/4.
# DRAMA 4.1.4 SARA uses the same model: reproduced to machine precision (tests/test_casualty.py).
CASUALTY_HUMAN_AREA_M2 = 0.36
# Two DRAMA 4.1.4 SARA runs by the user (run mode "Reentry-risk", no Monte Carlo), Eaglet II-like
# satellite: 25 kg (EAGLET_MASS_KG), box EAGLET_BOX_M, tumbling, uncontrolled re-entry from a
# circular orbit, osculating a = 6510 km, e = 0.001, i = 97.4 deg, start 2030-04-27,
# 1D projection with inclinationAngle = 97.4 deg, population UN WPP 2022 MEDIUM-VARIANT,
# casualty threshold 15 J. Read at run time, never copied (skipped if missing).
# ASSUMPTION (both inventories): no Eaglet II internal layout is public.
SARA_RUNS = {
    "base": {
        "folder": "~/DRAMA_projects/EagletII_SARA",
        "text": "all AA7075 structure (optimistic, wall ~10 mm)"},
    "components": {
        "folder": "~/DRAMA_projects/EagletII_SARA_components",
        "text": "AA7075 structure 16 kg + 4 A316 wheels 0.5 kg, TiAl6V4 tank 1.5 kg, "
                "Zerodur-proxy optics 2 kg, battery 1.5 kg, electronics 2 kg (pessimistic)"},
}
SARA_OUTPUT_FILE = "output/dmf_output.json"  # inside each folder
REENTRY_INC_DEG = 97.4  # inclination of the SARA runs (IRIDE Eaglet II SSO, as the MASTER-8 runs)
# 1D population density vs latitude shipped with DRAMA 4.1.4 SARA (UN World Population Prospects
# 2022, re-distributed by ESA): step 1/120 deg, people/km^2, columns 2024-2100. Read at run time.
SARA_POPULATION_1D_PATH = ("/Applications/DRAMA-4.1.4/TOOLS/SARA/RISK/data/Population_Density_1D/"
                           "Population_Growth_1D_MEDIUM-VARIANT.csv")  # scenario of the SARA runs
POPULATION_YEAR = 2030  # year of the SARA re-entry epoch (2030-04-27); the column SARA used
POPULATION_YEARS_SWEEP = (2025, 2030, 2035, 2040, 2045)  # sensitivity: span of the re-entries

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
DISPOSAL_ALT_KM = 375.0  # ESSB-ST-U-007 5.4.2.3c (apogee < 375 km, large constellations only,
                         # 3.2.25: not required for Eaglet II, 24 satellites; used here as a
                         # conservative disposal orbit). ASSUMPTION: instantaneous
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

# --- Time-varying solar activity (decay.decay_trajectory, v0.3) ---
# Density at a given F10.7: interpolation between the three NRLMSISE-00 / JB-2006 tables
# above, at their F10.7 values (Table 6-3 long-term: low 65, moderate 140, high 250, page 45;
# Tables G-1..G-3 use F10.7 = F10.7avg at these values).
SOLAR_F107_NODES = (65.0, 140.0, 250.0)
# ASSUMPTION: ln(rho) linear in F10.7 between the nodes (reference); rho linear in F10.7 as
# sensitivity. Argument: ln(rho) is concave in F10.7 at every node 120-900 km in both models
# (checked in tests/test_solar_cycle.py), so if the true curve keeps this curvature the log
# interpolation gives a LOWER density -> LONGER lifetime (conservative for the 5-year rule) and
# the linear one an upper density. Outside 65-250: ValueError (no extrapolation).
# ASSUMPTION: Ap is interpolated together with F10.7 (Tables G-1..G-3 pair F10.7 = 65/140/250
# with Ap = 0/15/45), although Ap is only weakly correlated with F10.7 (6.2.2 NOTE 1).
SOLAR_INTERP_REF = "log"
SOLAR_INTERP_SENSITIVITY = "linear"
# Scenario "strong cycle": ECSS-E-ST-10-04C Rev.1 Table A-1 (solar cycle 23, 30-day months,
# pages 85-88), required for realistic index sequences by 0760023c (page 45). Cycle 23 is a
# strong cycle (F81 peak 224 sfu): it shortens the lifetimes, so it is NOT conservative for
# the 5-year rule.
SOLAR_CYCLE_CSV = "inputs/ecss_solar_cycle23.csv"
SOLAR_INDEX = "F81mean"              # 81-day average: input of the atmosphere models (6.2.2 NOTE 2)
SOLAR_INDEX_SENSITIVITY = "F10mean"  # monthly mean of the daily F10.7
SOLAR_MONTH_DAYS = 30.0              # Table A-1: 30-day (1 month) intervals
# ASSUMPTION: Table A-1 month 0 = start (minimum) of the cycle, and the cycle repeats with
# period 140 months (11.5 yr); needed because lifetimes exceed one cycle.
# Scenario "ESA forecast 2021 (fap_mon.dat)" (conservative, weak cycle 25): ESA long-term
# forecast in TOOLS/OSCAR/data/fap_mon.dat of DRAMA 4.1.4, header "Last Update: 2021 May 4", last
# definitive point 10/2020, prediction to 12/2129. Column used: predicted 13-month centred mean
# F10.7. Forecast cycle 25 peak ~146 sfu (2025). NOTE: the OBSERVED cycle 25 was strong: the
# observed series used by OSCAR "latest prediction" (OSCAR_S3B_RUNS below) has F3M peak 225 sfu
# on 2024-10-09, about as cycle 23 (224). This file was NOT used by those OSCAR runs. Read at run
# time, NEVER copied into this repository; if the file is missing the scenario is skipped.
# Calendar months (step = 1/12 yr), not periodic.
OSCAR_FORECAST_PATH = "/Applications/DRAMA-4.1.4/TOOLS/OSCAR/data/fap_mon.dat"
OSCAR_FORECAST_INFO = "DRAMA 4.1.4 fap_mon.dat, last update 2021-05-04"
# Start dates of the solar cycles (phase 0):
# cycle 24: 2008-12-01 = DRAMA 4.1.4 TOOLS/OSCAR/data/solcycle.lst (cycle 24 1 Dec 2008 - 31 Dec
#           2019); ECSS-E-ST-10-04C Rev.1 Table B-2 gives the cycle 24 minimum at 2008.9 (agrees).
# cycle 25: 2020-01-01 = DRAMA solcycle.lst (end of cycle 24 on 2019-12-31). NOT in the ECSS
#           (Table B-2 stops at the cycle 24 minimum; formula B-1 gives maxima only). The ECSS-only
#           alternative 2008.9 + 11.5 yr (Table A-1 length) = 2020.4 is used as sensitivity.
#           The forecast file has its minimum F10.7 in 12/2019 (consistent).
CYCLE24_START = "2008-12-01"
CYCLE25_START = "2020-01-01"
CYCLE25_START_ECSS_ONLY = "2020-06-01"  # 2008-12-01 + 140 x 30 days (sensitivity)
# Phase of the start of the mission (lifecycle time t = 0 = first launch):
# general case: ASSUMPTION uniform over one cycle, one draw per Monte Carlo run (launch date
# unknown with respect to the cycle); strong cycle 0-140 months; ESA forecast 2021 from the start
# of cycle 25 to the next minimum of the forecast (found in the file).
# IRIDE case: first launch of the Eaglet II satellites 2025-11-28. Source: ASI, "IRIDE - Satellite
# Constellation for Earth Observation", https://www.asi.it/en/?p=62498: 8 Eaglet II launched on
# 28 November 2025, SpaceX Falcon 9, Transporter-15, Vandenberg.
IRIDE_FIRST_LAUNCH = "2025-11-28"
SOLAR_PHASE_SEED_STREAM = 8  # seed sequence [SEED, 8] of the mission start phase (separate stream)
SOLAR_PHASE_GRID_MONTHS = 1.0  # spacing of the phase grid of the residual lifetime tables [months]
SOLAR_PHASE_GRID_TRADE_MONTHS = 3.0  # coarser grid in the altitude trade (run time; the residual
                                    # lifetime varies smoothly with the phase at 600-800 km)
# ESA forecast 2021: the file ends in 12/2129; where the decay is not over by then (800 km) the
# lifetime is reported as a LOWER bound ('>='), no extrapolation.

# --- Reference case for decay.py: DRAMA 4.1.4 OSCAR, Sentinel-3B (three runs by the user) ---
# OSCAR v2.5.2, FOCUS v2.6.2, geopotential GEM-T1 (6x6), drag NRLMSISE-00 (PyOscar.oop of each
# run). Input (input/dmf_input.json): initial orbit a = 7186.84 km, e = 0.00085, i = 98.62 deg
# (mean elements), disposal "Circular Delayed De-Orbit", lifetime limit 15 yr, lifetime margin
# 5 %, 1 Hohmann transfer. OSCAR searches the circular disposal altitude (bisection) whose
# propagated lifetime matches the limit, then gives delta-v and propellant.
# Lifetime margin: DRAMA 4.1.4 documentation, docu/software_operation/single_analysis_tools/
# OSCAR/oscar_operation.html (HTML, no page numbers), section on the disposal options: "The
# Lifetime margin accounts for force model and integration errors. A recommended value by ISO
# 27852:2024 is 5 %". ISO 27852 itself not read. ASSUMPTION on how it is applied (not stated in
# the documentation): the outputs are compatible both with a target 15 / 1.05 = 14.286 yr and
# with 15 x 0.95 = 14.25 yr (14.283 is closer to the first, 14.259 to the second); with the
# bisection tolerance the two cannot be told apart. The comparison uses the PROPAGATED lifetime
# of each run (= final date - start), so it does not depend on this choice.
S3B_INC_DEG = 98.62
S3B_AREA_M2 = 16.435
S3B_MASS_KG = 1130.0
S3B_START = "2018-04-26T15:44"   # start of the OSCAR runs (UTC)
S3B_SMA_INITIAL_KM = 7186.84     # initial semi-major axis (input), for the Hohmann delta-v check
S3B_ISP_S = 200.0                # input/dmf_input.json "specificImpulse": 200 (disposal system
                                 # "Chemical Propulsion"), for the propellant check (main.py)
# One entry per run, values from output/dmf_output.json (PyOscar.oop) of each folder; solar data
# files from logs/Oscar-*.log. "activity": input of our model for the same comparison.
OSCAR_S3B_RUNS = {
    "constant moderate": {
        "folder": "~/DRAMA_projects/S3B_constant_moderate",
        "solar_option": "Constant F10.7 = 140, Ap = 15",
        "solar_files": "none (constant indices)",
        "activity": "constant moderate",   # ECSS Table G-2: F10.7 = F10.7avg = 140, Ap = 15 (same)
        "perigee_km": 634.05, "lifetime_years": 14.259, "reentry": "2032-07-29 22:15:16",
        "delta_v_m_s": 92.17, "propellant_kg": 54.35},
    "ECSS cycle 23": {
        "folder": "~/DRAMA_projects/S3B_ECSS_cycle23",
        "solar_option": "ECSS sample solar cycle (cycle 23 repeated, start 2008-12-01 in the log)",
        "solar_files": "DRAMA TOOLS/OSCAR/data/ecss.dat",
        "activity": "strong cycle",        # our Table A-1 F81mean, phase from CYCLE24_START
        "perigee_km": 614.69, "lifetime_years": 14.283, "reentry": "2032-08-08 00:19:52",
        "delta_v_m_s": 102.59, "propellant_kg": 60.66},
    "latest prediction": {
        "folder": "~/DRAMA_projects/DRAMA_example",
        "solar_option": "Latest prediction (ESA data), covered by observed data, no forecast",
        "solar_files": "ESA daily file of 2025-09-17, ESA monthly file of 2025-08-25 (log)",
        "activity": "OSCAR series",        # PyOscar.osg F3M, monthly means (read at run time)
        "perigee_km": 600.27, "lifetime_years": 14.283, "reentry": "2032-08-08 00:58:58",
        "delta_v_m_s": 110.39, "propellant_kg": 65.40},
}
OSCAR_OUTPUT_FILE = "output/dmf_output.json"  # inside each folder; read at run time, never copied
# Solar series used by OSCAR (PyOscar.osg, daily: date, F10, F3M = 81-day mean, Ap). Our model
# takes the monthly mean of F3M (calendar months), as F81mean for the strong cycle.
# ECSS run checked (main.py): F3M vs our F81mean 2018-2032 mean difference -0.6 sfu, mean |diff|
# 3.8 sfu (1.6 sfu if shifted by ~25 days: different 81-day alignment); Ap in OSCAR is a STEP of
# the daily F10 (F10 < 100 -> 0, 100-199 -> 15, >= 200 -> 39), ours is continuous in F81 (0/15/45).

# --- Simulation ---
SIM_DURATION_YEARS = 15.0  # ASSUMPTION: reference mission horizon, to be reviewed
DT_DAYS = 1.0              # fixed time step (project rule, CLAUDE.md)
N_RUNS = 1000              # number of Monte Carlo runs
N_RUNS_SWEEP = 1000        # Monte Carlo runs per point of the parameter sweeps in main.py
# N_RUNS and N_RUNS_SWEEP above are the values of the report (python main.py --full).
N_RUNS_FAST = 200          # ASSUMPTION: runs of the default fast mode of main.py (both N_RUNS and
                           # N_RUNS_SWEEP), a compromise run time / precision: results approximate
SEED = 42                  # master seed of the Monte Carlo
