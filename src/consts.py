P0 = 1.0e+5
T0 = 20.0
P_ATM = 1.0e+5
T_ATM = 20.0

GRAVITY = 9.81

POROSITY = 0.10
CONDUCTIVITY = 2.5
DENSITY = 2.5e+3
SPECIFIC_HEAT = 1.0e+3

MASS_ENTHALPY = 1.5e+6
HEAT_RATE = 0.2

MAX_NS_TSTEPS = 500
MAX_PR_TSTEPS = 200
NS_STEPSIZE = 1.0e+15

MSG_END_TIME = ["info", "timestep", "end_time_reached"]
MSG_MAX_STEP = ["info", "timestep", "stop_size_maximum_reached"]
MSG_MAX_ITS = ["info", "timestep", "max_timesteps_reached"]
MSG_ABORTED = ["warn", "timestep", "aborted"]

SECS_PER_WEEK = 60.0 ** 2 * 24.0 * 7.0

NESI_WAI_PATH = "/nesi/project/uoa00463/bin/waiwera"
NESI_OPTIONS = "--exclusive --exact --ntasks=1 --nodes=1 --cpus-per-task=1"