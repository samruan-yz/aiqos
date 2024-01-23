#!/usr/bin/python

import os
import time
import shlex
import numpy as np
import random as rd
import subprocess as sp
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
import sklearn.gaussian_process as gp
import subprocess
import sys

# Number of LC apps available
TOT_LC_APPS = 5

# Number of BG apps available
TOT_BG_APPS = 6

# Number of QPS categories
N_QPS_CAT = 10

# LC apps
APP_NAMES = ["img-dnn", "masstree", "memcached", "specjbb", "xapian"]

# QoS requirements of LC apps (time in seconds)
APP_QOSES = {
    "img-dnn": 3.0,
    "masstree": 2.0,
    "memcached": 225.0,
    "specjbb": 0.5,
    "xapian": 12.0,
}
# QPS levels
APP_QPSES = {
    "img-dnn": list(range(300, 3300, 300)),
    "masstree": list(range(100, 1100, 100)),
    "memcached": list(range(20000, 220000, 20000)),
    "specjbb": list(range(800, 9600, 800)),
    "xapian": list(range(800, 9600, 800)),
}

# BG apps
BCKGRND_APPS = [
    "blackscholes",
    "canneal",
    "fluidanimate",
    "freqmine",
    "streamcluster",
    "swaptions",
]

# Number of times acquisition function optimization is restarted
NUM_RESTARTS = 1

# Number of maximum iterations (max configurations sampled)
MAX_ITERS = 20

# Shared Resources hardware configuration:
# Number of Cores (10 units)
# Number of Ways (11 units)
# Percent Memory Bandwidth (10 units)

# Number of resources controlled
NUM_RESOURCES = 3

# Max values of each resources
NUM_CORES = 24
NUM_WAYS = 11
MEMORY_BW = 100

# Max units of (cores, LLC ways, memory bandwidth)
NUM_UNITS = [24, 11, 10]

# Configuration formats
CONFIGS_CORES = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
]
CONFIGS_CWAYS = [
    "0x001",
    "0x003",
    "0x007",
    "0x00f",
    "0x01f",
    "0x03f",
    "0x07f",
    "0x0ff",
    "0x1ff",
    "0x3ff",
    "0x7ff",
]
CONFIGS_MEMBW = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100"]

# Commands to set hardware allocations
TASKSET = "sudo taskset -acp "
COS_CAT_SET1 = 'sudo pqos -e "llc:%s=%s"'
COS_CAT_SET2 = 'sudo pqos -a "llc:%s=%s"'
COS_MBG_SET1 = 'sudo pqos -e "mba:%s=%s"'
COS_MBG_SET2 = 'sudo pqos -a "core:%s=%s"'
COS_RESET = "sudo pqos -R"

# Commands to get MSRs
WR_MSR_COMM = "wrmsr -a "
RD_MSR_COMM = "rdmsr -a -u "

# MSR register requirements
IA32_PERF_GBL_CTR = "0x38F"  # Need bits 34-32 to be 1
IA32_PERF_FX_CTRL = "0x38D"  # Need bits to be 0xFFF
MSR_PERF_FIX_CTR0 = "0x309"

# Amount of time to sleep after each sample
SLEEP_TIME = 2

# Suppress application outputs
FNULL = open(os.devnull, "w")

# Path to the base directory (if required)
BASE_DIR = "/path/to/base/directory/"

# All the LC apps being run
LC_APPS = ["resnet", "bert"]

# Path to the latency files of applications
LATS_FILES = [
    BASE_DIR + "tailbench-v0.9/img-dnn/lats.bin",
    BASE_DIR + "tailbench-v0.9/xapian/lats.bin",
]

# ALl the BG jobs being runs
BG_APPS = []

APPS = LC_APPS + BG_APPS

# PIDs of all the applications in order of APPS
APP_PIDS = ["11475", "11783"]

# QoSes of LC apps
# APP_QOSES = [APP_QOSES[a] for a in LC_APPS]
APP_QOSES = [15.000000, 130.000000]

# Number of apps currently running
NUM_LC_APPS = len(LC_APPS)

NUM_BG_APPS = len(BG_APPS)

NUM_APPS = NUM_LC_APPS + NUM_BG_APPS

# Total number of parameters
NUM_PARAMS = NUM_RESOURCES * (NUM_APPS - 1)

# Set expected value threshold for termination
# EI_THRESHOLD = 0.01**NUM_APPS
EI_THRESHOLD = 0

# Global variable to hold baseline performances
BASE_PERFS = [0.0] * NUM_APPS

# Required global variables
BOUNDS = None

CONSTS = None

MODEL = None

OPTIMAL_PERF = None

RESNET_QPS = 27
BERT_QPS = 7

if len(sys.argv) < 3:
    print("Usage: script.py param1 param2")
    sys.exit(1)
RESNET_QPS = sys.argv[1]
BERT_QPS = sys.argv[2]
print(RESNET_QPS + " " + BERT_QPS)

# Added global variables
LATENCY_FILE = f"/data3/ydil/aiqos/inference_results_v1.1/closed/Intel/code/data_point/resnet_bert_search/result_{RESNET_QPS}_{BERT_QPS}.data"
RUN_SCRIPT = "collocation_run_resnet_bert.sh"
MAIN_LOG_FILE = f"/data3/ydil/aiqos/inference_results_v1.1/closed/Intel/code/CLITE_LOG/clite_result_{RESNET_QPS}_{BERT_QPS}.data"
RUN_SCRIPT_LOG_FILE = f"/data3/ydil/aiqos/inference_results_v1.1/closed/Intel/code/CLITE_LOG/run_script_log_{RESNET_QPS}_{BERT_QPS}.data"
ONLINE_MODE = True


# Added helper functions
def set_cores(cores):
    with open(RUN_SCRIPT, "r") as file:
        lines = file.readlines()

    # Current core end
    current_end = -1

    # Modify the script using CORES
    for i in range(0, NUM_APPS):
        app_name = APPS[i]
        cores_range = cores[i]

        # Calculate the current core start according to the previous core end
        current_start = current_end + 1
        current_end = current_start + len(cores_range) - 1

        for idx, line in enumerate(lines):
            if line.startswith(app_name + "_core_start"):
                lines[idx] = f"{app_name}_core_start={current_start}\n"
            elif line.startswith(app_name + "_core_end"):
                lines[idx] = f"{app_name}_core_end={current_end}\n"

        # print("Change Core of", APP[i], ": ", len(CORES[i]), "cores")

    # Write back
    with open(RUN_SCRIPT, "w") as file:
        file.writelines(lines)


def set_cways(cways):
    with open(RUN_SCRIPT, "r") as file:
        lines = file.readlines()

    for i in range(0, NUM_APPS):
        app_name = APPS[i]

        # Update script based on the WAYS values
        llc_hex = cways[i]

        for idx, line in enumerate(lines):
            if line.startswith(app_name + "_llc"):
                lines[idx] = f"{app_name}_llc={llc_hex}\n"

    # Write the modified content back to the file
    with open(RUN_SCRIPT, "w") as file:
        file.writelines(lines)


def set_mbs(membw):
    with open(RUN_SCRIPT, "r") as file:
        lines = file.readlines()

    for i in range(0, NUM_APPS):
        app_name = APPS[i]

        # Update script based on the WAYS values
        bw = membw[i]

        for idx, line in enumerate(lines):
            if line.startswith(app_name + "_mb"):
                lines[idx] = f"{app_name}_mb={bw}\n"

    # Write the modified content back to the file
    with open(RUN_SCRIPT, "w") as file:
        file.writelines(lines)


def get_lat(index):
    with open(LATENCY_FILE, "r") as file:
        # Skip the first and second line (caption)
        file.readline()
        file.readline()

        while index != 0:
            file.readline()
            index -= 1

        # Extract values for the given number of lines
        line = file.readline().strip()
        values = line.split()
        # Convert to float, and store in the dictionary
        latency = float(values[13])
        return latency


def print_initial_connfig(x, lats, ratio, y, idx):
    global MAIN_LOG_FILE

    with open(MAIN_LOG_FILE, "a") as log_file:
        log_file.write(f"------Result for initial config {idx}------\n")
        log_file.write(
            "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
                "Task", "Cores", "LLCways", "MemBW", "QoS", "Ratio"
            )
        )

        app1 = f"{APPS[0]:<10} {CONFIGS_CORES[x[0]-1]:<10} {x[1]:<10} {CONFIGS_MEMBW[x[2]-1]:<10} {lats[0]:<10.2f} {ratio[0]:<10.2f}\n"
        app2 = f"{APPS[1]:<10} {CONFIGS_CORES[24-x[0]-1]:<10} {11-x[1]:<10} {CONFIGS_MEMBW[10-x[2]-1]:<10} {lats[1]:<10.2f} {ratio[1]:<10.2f}\n"

        log_file.write(app1)
        log_file.write(app2)
        log_file.write(f"Performance score for this config: {y}\n\n")


# x = configuration, q = QoS, y = performance score
def print_res(x, lats, ratio, y, round, ei):
    global MAIN_LOG_FILE

    with open(MAIN_LOG_FILE, "a") as log_file:
        log_file.write(f"------Result for round {round}------\n")
        log_file.write(
            "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(
                "Task", "Cores", "LLCways", "MemBW", "QoS", "Ratio"
            )
        )

        app1 = f"{APPS[0]:<10} {CONFIGS_CORES[x[0]-1]:<10} {x[1]:<10} {CONFIGS_MEMBW[x[2]-1]:<10} {lats[0]:<10.2f} {ratio[0]:<10.2f}\n"
        app2 = f"{APPS[1]:<10} {CONFIGS_CORES[24-x[0]-1]:<10} {11-x[1]:<10} {CONFIGS_MEMBW[10-x[2]-1]:<10} {lats[1]:<10.2f} {ratio[1]:<10.2f}\n"

        log_file.write(app1)
        log_file.write(app2)
        log_file.write(f"Performance score for this config: {y}\n")
        log_file.write(f"Expected improvement for this config: {ei}\n\n")


def log_run_script(idx):
    global RUN_SCRIPT
    global RUN_SCRIPT_LOG_FILE

    with open(RUN_SCRIPT, "r") as script_file:
        script_content = script_file.read()

    with open(RUN_SCRIPT_LOG_FILE, "a") as log_file:
        log_file.write(f"------ Log for round {idx} ------\n")
        log_file.write(script_content)
        log_file.write("\n\n")


# This class is used to parse latency files and extract request times, service times, and sojourn times.
class Lat(object):
    def __init__(self, fileName):
        f = open(fileName, "rb")
        a = np.fromfile(f, dtype=np.uint64)
        self.reqTimes = a.reshape((int(a.shape[0] / 3.0), 3))
        f.close()

    def parseQueueTimes(self):
        return self.reqTimes[:, 0]

    def parseSvcTimes(self):
        return self.reqTimes[:, 1]

    def parseSojournTimes(self):
        return self.reqTimes[:, 2]


# Computes the 95th percentile of sojourn times from a given latency file.
def getLatPct(latsFile):
    assert os.path.exists(latsFile)

    latsObj = Lat(latsFile)

    sjrnTimes = [l / 1e6 for l in latsObj.parseSojournTimes()]

    mnLt = np.mean(sjrnTimes)

    p95 = stats.scoreatpercentile(sjrnTimes, 95.0)

    return p95


# Generates bounds and constraints for the optimizer.
def gen_bounds_and_constraints():
    global BOUNDS, CONSTS

    # Generate the bounds and constraints required for the optimizer
    BOUNDS = (
        np.array(
            [
                [[1, NUM_UNITS[r] - (NUM_APPS - 1)] for a in range(NUM_APPS - 1)]
                for r in range(NUM_RESOURCES)
            ]
        )
        .reshape(NUM_PARAMS, 2)
        .tolist()
    )

    CONSTS = []
    for r in range(NUM_RESOURCES):
        CONSTS.append(
            {
                "type": "eq",
                "fun": lambda x: sum(x[r * (NUM_APPS - 1) : (r + 1) * (NUM_APPS - 1)])
                - (NUM_APPS - 1),
            }
        )
        CONSTS.append(
            {
                "type": "eq",
                "fun": lambda x: -sum(x[r * (NUM_APPS - 1) : (r + 1) * (NUM_APPS - 1)])
                + (NUM_UNITS[r] - 1),
            }
        )


# Generates initial configurations for resource partitioning.
def gen_initial_configs():
    # Generate the maximum allocation configurations for all applications
    configs = [[1] * NUM_PARAMS for j in range(NUM_APPS)]
    for j in range(NUM_APPS - 1):
        for r in range(NUM_RESOURCES):
            configs[j][j + ((NUM_APPS - 1) * r)] = NUM_UNITS[r] - (NUM_APPS - 1)

    # Generate the equal partition configuration
    equal_partition = []
    for r in range(NUM_RESOURCES):
        for j in range(NUM_APPS - 1):
            equal_partition.append(int(NUM_UNITS[r] / NUM_APPS))
    configs.append(equal_partition)

    return configs


# Computes baseline performances for given configurations.
def get_baseline_perfs(configs):
    global BASE_PERFS

    for i in range(NUM_APPS):
        p = configs[i]

        # Core allocations of each job
        app_cores = [""] * NUM_APPS
        s = 0
        for j in range(NUM_APPS - 1):
            # app_cores[j] = ",".join(
            #     [
            #         str(c)
            #         for c in list(range(s, s + p[j]))
            #         + list(range(s + NUM_UNITS[0], s + p[j] + NUM_UNITS[0]))
            #     ]
            # )
            app_cores[j] = [str(c) for c in list(range(s, s + p[j]))]
            s += p[j]
        # app_cores[NUM_APPS - 1] = ",".join(
        #     [
        #         str(c)
        #         for c in list(range(s, NUM_UNITS[0]))
        #         + list(range(s + NUM_UNITS[0], NUM_UNITS[0] + NUM_UNITS[0]))
        #     ]
        # )
        app_cores[NUM_APPS - 1] = [str(c) for c in list(range(s, NUM_UNITS[0]))]

        # L3 cache ways allocation of each job
        app_cways = [""] * NUM_APPS
        s = 0
        for j in range(NUM_APPS - 1):
            app_cways[j] = str(
                hex(
                    int(
                        "".join(
                            [str(1) for w in list(range(p[j + NUM_APPS - 1]))]
                            + [str(0) for w in list(range(s))]
                        ),
                        2,
                    )
                )
            )
            s += p[j + NUM_APPS - 1]
        app_cways[NUM_APPS - 1] = str(
            hex(
                int(
                    "".join(
                        [str(1) for w in list(range(NUM_UNITS[1] - s))]
                        + [str(0) for w in list(range(s))]
                    ),
                    2,
                )
            )
        )

        # Memory bandwidth allocation of each job
        app_membw = [""] * NUM_APPS
        s = 0
        for j in range(NUM_APPS - 1):
            app_membw[j] = str(p[j + 2 * (NUM_APPS - 1)] * 10)
            s += p[j + 2 * (NUM_APPS - 1)] * 10
        app_membw[NUM_APPS - 1] = str(NUM_UNITS[2] * 10 - s)

        # Set the allocations
        # for j in range(NUM_APPS):
        #     taskset_cmnd = TASKSET + app_cores[j] + " " + APP_PIDS[j]
        #     cos_cat_set1 = COS_CAT_SET1 % (str(j + 1), app_cways[j])
        #     cos_cat_set2 = COS_CAT_SET2 % (str(j + 1), app_cores[j])
        #     cos_mBG_set1 = COS_MBG_SET1 % (str(j + 1), app_membw[j])
        #     cos_mBG_set2 = COS_MBG_SET2 % (str(j + 1), app_cores[j])
        #     sp.check_output(shlex.split(taskset_cmnd), stderr=FNULL)
        #     sp.check_output(shlex.split(cos_cat_set1), stderr=FNULL)
        #     sp.check_output(shlex.split(cos_cat_set2), stderr=FNULL)
        #     sp.check_output(shlex.split(cos_mBG_set1), stderr=FNULL)
        #     sp.check_output(shlex.split(cos_mBG_set2), stderr=FNULL)
        set_cores(app_cores)
        set_cways(app_cways)
        set_mbs(app_membw)

        # if i >= NUM_LC_APPS:
        #     # Reset the IPS counters
        #     os.system(WR_MSR_COMM + MSR_PERF_FIX_CTR0 + " 0x0")

        if ONLINE_MODE:
            subprocess.call(["bash", RUN_SCRIPT])
        else:
            time.sleep(SLEEP_TIME)
        # Run script here
        # subprocess.call(["bash", RUN_SCRIPT])

        if i < NUM_LC_APPS:
            BASE_PERFS[i] = get_lat(i)
        else:
            # Get the IPS counters
            ipsP = os.popen(RD_MSR_COMM + MSR_PERF_FIX_CTR0)

            # Calculate the IPS
            IPS = 0.0
            cor = [int(c) for c in app_cores[i].split(",")]
            ind = 0
            for line in ipsP.readlines():
                if ind in cor:
                    IPS += float(line)
                ind += 1

            BASE_PERFS[i] = IPS


# Generates a random configuration for resource partitioning.
def gen_random_config():
    # Generate a random configuration
    config = []
    for r in range(NUM_RESOURCES):
        total = 0
        remain_apps = NUM_APPS
        for j in range(NUM_APPS - 1):
            alloc = rd.randint(1, NUM_UNITS[r] - (total + remain_apps - 1))
            config.append(alloc)
            total += alloc
            remain_apps -= 1

    return config


# Samples the performance of a given configuration.
def sample_perf(p):
    # Core allocations of each job
    app_cores = [""] * NUM_APPS
    s = 0
    for j in range(NUM_APPS - 1):
        # app_cores[j] = ",".join(
        #     [
        #         str(c)
        #         for c in list(range(s, s + p[j]))
        #         + list(range(s + NUM_UNITS[0], s + p[j] + NUM_UNITS[0]))
        #     ]
        # )
        app_cores[j] = [str(c) for c in list(range(s, s + p[j]))]
        s += p[j]
    # app_cores[NUM_APPS - 1] = ",".join(
    #     [
    #         str(c)
    #         for c in list(range(s, NUM_UNITS[0]))
    #         + list(range(s + NUM_UNITS[0], NUM_UNITS[0] + NUM_UNITS[0]))
    #     ]
    # )
    app_cores[NUM_APPS - 1] = [str(c) for c in list(range(s, NUM_UNITS[0]))]

    # L3 cache ways allocation of each job
    app_cways = [""] * NUM_APPS
    s = 0
    for j in range(NUM_APPS - 1):
        app_cways[j] = str(
            hex(
                int(
                    "".join(
                        [str(1) for w in list(range(p[j + NUM_APPS - 1]))]
                        + [str(0) for w in list(range(s))]
                    ),
                    2,
                )
            )
        )
        s += p[j + NUM_APPS - 1]
    app_cways[NUM_APPS - 1] = str(
        hex(
            int(
                "".join(
                    [str(1) for w in list(range(NUM_UNITS[1] - s))]
                    + [str(0) for w in list(range(s))]
                ),
                2,
            )
        )
    )

    # Memory bandwidth allocation of each job
    app_membw = [""] * NUM_APPS
    s = 0
    for j in range(NUM_APPS - 1):
        app_membw[j] = str(p[j + 2 * (NUM_APPS - 1)] * 10)
        s += p[j + 2 * (NUM_APPS - 1)] * 10
    app_membw[NUM_APPS - 1] = str(NUM_UNITS[2] * 10 - s)

    # Set the allocations
    # for j in range(NUM_APPS):
    #     taskset_cmnd = TASKSET + app_cores[j] + " " + APP_PIDS[j]
    #     cos_cat_set1 = COS_CAT_SET1 % (str(j + 1), app_cways[j])
    #     cos_cat_set2 = COS_CAT_SET2 % (str(j + 1), app_cores[j])
    #     cos_mBG_set1 = COS_MBG_SET1 % (str(j + 1), app_membw[j])
    #     cos_mBG_set2 = COS_MBG_SET2 % (str(j + 1), app_cores[j])
    #     sp.check_output(shlex.split(taskset_cmnd), stderr=FNULL)
    #     sp.check_output(shlex.split(cos_cat_set1), stderr=FNULL)
    #     sp.check_output(shlex.split(cos_cat_set2), stderr=FNULL)
    #     sp.check_output(shlex.split(cos_mBG_set1), stderr=FNULL)
    #     sp.check_output(shlex.split(cos_mBG_set2), stderr=FNULL)
    set_cores(app_cores)
    set_cways(app_cways)
    set_mbs(app_membw)

    if NUM_BG_APPS != 0:
        # Reset the IPS counters
        os.system(WR_MSR_COMM + MSR_PERF_FIX_CTR0 + " 0x0")

    # Wait for some cycles
    if ONLINE_MODE:
        subprocess.call(["bash", RUN_SCRIPT])
    else:
        time.sleep(SLEEP_TIME)
    # Run script here
    # subprocess.call(["bash", RUN_SCRIPT])

    # If QoS met, qv = [1.0, 1.0, 1.0] = sd
    # If QoS not met, qv = [0.5, 0.5, 0.5], sd = [0.25, 0.25, 0.25]
    # QoS values
    qv = [1.0] * NUM_LC_APPS
    # Performance scores
    sd = [1.0] * NUM_LC_APPS
    # Lantencys
    lats = [1.0] * NUM_LC_APPS
    for j in range(NUM_LC_APPS):
        # p95 = getLatPct(LATS_FILES[j])
        p99 = get_lat(j)
        lats[j] = p99
        if p99 > APP_QOSES[j]:
            qv[j] = APP_QOSES[j] / p99
            sd[j] = BASE_PERFS[j] / p99

    # Return the final objective function score if QoS not met
    if stats.mstats.gmean(qv) != 1.0:
        return qv, 0.5 * stats.mstats.gmean(qv), lats

    # Return the final objective function score if QoS met
    if NUM_BG_APPS == 0:
        return qv, 0.5 * (min(1.0, stats.mstats.gmean(sd)) + 1.0), lats

    # Get the IPS counters
    ipsP = os.popen(RD_MSR_COMM + MSR_PERF_FIX_CTR0)

    sd = [0.0] * NUM_BG_APPS
    for j in range(NUM_BG_APPS):
        # Calculate the IPS
        IPS = 0.0
        cor = [int(c) for c in app_cores[j + NUM_LC_APPS].split(",")]
        ind = 0
        for line in ipsP.readlines():
            if ind in cor:
                IPS += float(line)
            ind += 1

        sd[j] = min(1.0, IPS / BASE_PERFS[j + NUM_LC_APPS])

    # Return the final objective function score if BG jobs are present
    return qv, 0.5 * (min(1.0, stats.mstats.gmean(sd)) + 1.0), lats


# Computes the expected improvement for a given configuration.
def expected_improvement(c, exp=0.01):
    # Calculate the expected improvement for a given configuration 'c'
    mu, sigma = MODEL.predict(np.array(c).reshape(-1, NUM_PARAMS), return_std=True)
    val = 0.0
    with np.errstate(divide="ignore"):
        Z = (mu - OPTIMAL_PERF - exp) / sigma
        val = (mu - OPTIMAL_PERF - exp) * norm.cdf(Z) + sigma * norm.pdf(Z)
        val[sigma == 0.0] = 0.0

    return -1 * val


# Finds the next configuration to sample based on expected improvement.
def find_next_sample(x, q, y):
    # Generate the configuration which has the highest expected improvement potential
    max_config = None
    max_result = 1

    # Multiple restarts to find the global optimum of the acquisition function
    for n in range(NUM_RESTARTS):
        val = None

        # Perform dropout 1/4 of the times
        if rd.choice([True, True, True, False]):
            x0 = gen_random_config()

            val = minimize(
                fun=expected_improvement,
                x0=x0,
                bounds=BOUNDS,
                constraints=CONSTS,
                method="SLSQP",
            )
        else:
            ind = rd.choice(list(range(len(y))))
            app = q[ind].index(max(q[ind]))

            if app == (NUM_APPS - 1):
                consts = []
                for r in range(NUM_RESOURCES):
                    units = sum(x[ind][r * (NUM_APPS - 1) : (r + 1) * (NUM_APPS - 1)])
                    consts.append(
                        {
                            "type": "eq",
                            "fun": lambda x: sum(
                                x[r * (NUM_APPS - 1) : (r + 1) * (NUM_APPS - 1)]
                            )
                            - units,
                        }
                    )
                    consts.append(
                        {
                            "type": "eq",
                            "fun": lambda x: -sum(
                                x[r * (NUM_APPS - 1) : (r + 1) * (NUM_APPS - 1)]
                            )
                            + units,
                        }
                    )

                val = minimize(
                    fun=expected_improvement,
                    x0=x[ind],
                    bounds=BOUNDS,
                    constraints=consts,
                    method="SLSQP",
                )

            else:
                bounds = [[b[0], b[1]] for b in BOUNDS]

                for r in range(NUM_RESOURCES):
                    bounds[app + r * (NUM_APPS - 1)][0] = x[ind][
                        app + r * (NUM_APPS - 1)
                    ]
                    bounds[app + r * (NUM_APPS - 1)][1] = x[ind][
                        app + r * (NUM_APPS - 1)
                    ]

                val = minimize(
                    fun=expected_improvement,
                    x0=x[ind],
                    bounds=bounds,
                    constraints=CONSTS,
                    method="SLSQP",
                )

        if val.fun < max_result:
            max_config = val.x
            max_result = val.fun

    return -max_result, [int(c) for c in max_config]


# The main Bayesian Optimization engine that iteratively samples configurations to find the optimal one.
def bayesian_optimization_engine(x0, alpha=1e-5):
    global MODEL, OPTIMAL_PERF

    # Configurations
    x_list = []
    # QoS lists
    q_list = []
    # Performance scores
    # A performance score of 1 would mean that all applications are achieving their
    # best possible performance (i.e., meeting their QoS requirements for LC apps
    # and achieving maximum throughput for BG apps).
    y_list = []

    # Sample initial configurations
    i = 0
    for params in x0:
        x_list.append(params)
        q, y, lats = sample_perf(params)
        q_list.append(q)
        y_list.append(y)
        print_initial_connfig(params, lats, q, y, i)
        i += 1

    # Arraynize configuration list
    xp = np.array(x_list)
    # Arraynize performance score list
    yp = np.array(y_list)

    # Create the Gaussian process model as the surrogate model
    kernel = gp.kernels.Matern(length_scale=1.0, nu=1.5)
    MODEL = gp.GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=True
    )

    # Iterate for specified number of iterations as maximum
    for n in range(MAX_ITERS):
        # Update the surrogate model
        MODEL.fit(xp, yp)
        OPTIMAL_PERF = np.max(yp)

        # Find the next configuration to sample
        ei, next_sample = find_next_sample(x_list, q_list, y_list)

        # If the configuration is already s
        # ampled, carefully replace the sample
        mind = 0
        while next_sample in x_list:
            if mind == len(y_list):
                next_sample = gen_random_config()
                continue
            ind = sorted(enumerate(y_list), key=lambda x: x[1])[mind][0]
            if stats.mstats.gmean(q_list[ind]) == 1.0:
                mind += 1
                continue
            boxes = sum([q == 1.0 for q in q_list[ind]])
            if boxes == 0:
                mind += 1
                continue
            next_sample = [x for x in x_list[ind]]
            for r in range(NUM_RESOURCES):
                avail = NUM_UNITS[r]
                for a in range(NUM_APPS - 1):
                    if q_list[ind][a] == 1.0:
                        flip = rd.choice([True, False])
                        if flip and next_sample[r * (NUM_APPS - 1) + a] != 1.0:
                            next_sample[r * (NUM_APPS - 1) + a] -= 1
                        avail -= next_sample[r * (NUM_APPS - 1) + a]
                if q_list[ind][NUM_APPS - 1] == 1.0:
                    flip = rd.choice([True, False])
                    unit = NUM_UNITS[r] - sum(
                        next_sample[r * (NUM_APPS - 1) : (r + 1) * (NUM_APPS - 1)]
                    )
                    if flip and unit != 1.0:
                        avail -= unit - 1
                    else:
                        avail -= unit
                cnf = [
                    int(float(avail) / float(NUM_APPS - boxes))
                    for b in range(NUM_APPS - boxes)
                ]
                cnf[-1] += avail - sum(cnf)
                i = 0
                for a in range(NUM_APPS - 1):
                    if q_list[ind][a] != 1.0:
                        next_sample[r * (NUM_APPS - 1) + a] = cnf[i]
                        i += 1
            mind += 1

        # Sample the new configuration
        x_list.append(next_sample)
        q, y, lats = sample_perf(next_sample)
        q_list.append(q)
        y_list.append(y)
        print_res(next_sample, lats, q, y, n, ei)
        
        if y> OPTIMAL_PERF:
            OPTIMAL_PERF = y
            optimal_config = next_sample

        log_run_script(n)
    

        xp = np.array(x_list)
        yp = np.array(y_list)

        # Terminate if the termination requirements are met
        # If EI_THRESHOLD were set to 0.5, it would mean that unless the expected
        # improvement from the best unsampled point is at least 0.5
        # (or 50% improvement over the current best), the optimization would stop.
        # In the provided script, EI_THRESHOLD is set to 0.01**NUM_APPS, which means
        # the threshold is set very low, especially if there are multiple apps.
        # This ensures that the optimization process continues until the potential
        # improvements are very minor.
        if ei < EI_THRESHOLD or np.max(yp) >= 0.99:
            break

    return n + 1, np.max(yp), optimal_config


def c_lite():
    # Generate the bounds and constraints required for optimization
    gen_bounds_and_constraints()

    # Generate the initial set of configurations
    init_configs = gen_initial_configs()

    # Get the baseline performances with maximum allocations for each application
    get_baseline_perfs(init_configs)

    # Perform Bayesian optimization
    num_iters, obj_value, optimal_config = bayesian_optimization_engine(x0=init_configs)

    return num_iters, obj_value, optimal_config


def changeQPS():
    with open(RUN_SCRIPT, "r") as file:
        script_content = file.readlines()

    with open(RUN_SCRIPT, "w") as file:
        for line in script_content:
            if "for resnet_qps in" in line:
                file.write(f"for resnet_qps in {RESNET_QPS}\n")
            elif "for bert_qps in" in line:
                file.write(f"for bert_qps in {BERT_QPS}\n")
            else:
                file.write(line)


def main():
    changeQPS()
    # Switch on the performance counters
    # os.system(WR_MSR_COMM + IA32_PERF_GBL_CTR + " 0x70000000F")
    # os.system(WR_MSR_COMM + IA32_PERF_FX_CTRL + " 0xFFF")

    # Execute C-LITE
    num_iters, obj_value, optimal_config = c_lite()

    # Print the header
    st = ""
    for a in range(NUM_APPS):
        st += "App" + str(a) + ","
    st += "ObjectiveValue" + ","
    st += "#Iterations"

    print(st)
    log_file.write(st+ "\n")

    # Print the final results
    st = ""
    for a in LC_APPS:
        st += a + ","
    for a in BG_APPS:
        st += a + ","
    st += "%.2f" % obj_value + ","
    st += "%.2f" % num_iters + ","
    st += optimal_config

    print(st)
    log_file.write(st+ "\n")


if __name__ == "__main__":
    # Invoke the main function
    main()
