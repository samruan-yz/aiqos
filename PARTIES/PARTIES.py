#!/usr/bin/python
from time import sleep
import random
import sys
import os
import collections
from datetime import datetime

import subprocess

RESNET_QPS = 27
BERT_QPS = 7
ALL_VIOLATION_LIMIT = -0.1

if len(sys.argv) < 5:
    print("Usage: script.py param1 param2 core llc")
    sys.exit(1)
RESNET_QPS = sys.argv[1]
BERT_QPS = sys.argv[2]
TOTAL_CORES = int(sys.argv[3])
TOTAL_LLC = int(sys.argv[4])

CONFIG = "/data1/yufenggu/inference_spr/code/baselines/PARTIES/config.txt"  # default path to the input config.txt file
LATENCY_FILE = "/data1/yufenggu/inference_spr/code/data_point/resnet_bert_search/PARTIES_{}_{}.data".format(RESNET_QPS, BERT_QPS)
RUN_SCRIPT = "/data1/yufenggu/inference_spr/code/collocation_run_resnet_bert_PARTIES.sh"
LOG_FILE = "/data1/yufenggu/inference_spr/code/baselines/PARTIES/PARTIES_LOG/parties_resnet_bert_{}_{}.data".format(RESNET_QPS, BERT_QPS)

ROUND = 0
INTERVAL = 0.1  # Frequency of monitoring, unit is second
TIMELIMIT = 30  # How long to run this controller, unit is in second.
REST = 100
NUM = 0  # Number of colocated applications
APP = [None for i in range(10)]  # Application names
IP = [None for i in range(10)]  # IP of clients that run applications
QoS = [None for i in range(10)]  # Target QoS of each application
ECORES = [i for i in range(0, TOTAL_CORES, 1)]  # unallocated cores
CORES = [None for i in range(10)]  # CPU allocation
LOAD = []
FREQ = [2200 for i in range(10)]  # Frequency allocation
EWAY = 0  # unallocated ways
WAY = [0 for i in range(10)]  # Allocation of LLC ways
Lat = [0 for i in range(10)]  # Real-time tail latency
MLat = [0 for i in range(10)]  # Real-time tail latency of a moving window
Slack = [0 for i in range(10)]  # Real-time tail latency slack
LSlack = [0 for i in range(10)]  # Real-time tail latency slack in the last interval
LLSlack = [0 for i in range(10)]  # Real-time tail latency slack in the last interval
LDOWN = [0 for i in range(10)]  # Time to wait before this app can be downsized again
CPU = [0 for i in range(10)]  # CPU Utilization per core of each application
cCPU = collections.deque(maxlen=(int(5.0 / INTERVAL)))
MEM = [0 for i in range(10)]  # Total memory bandwidth usage of each application
State = [0 for i in range(10)]  # FSM State during resource adjustment
rLat = [[] for i in range(10)]  # Save real-time latency for final plotting
rrLat = [[] for i in range(10)]  # Save real-time latency for final plotting
rCORES = [[] for i in range(10)]  # Save real-time #cores for final plotting
rWAY = [[] for i in range(10)]  # Save real-time #ways for final plotting
rFREQ = [[] for i in range(10)]  # Save real-time frequency for final plotting
FF = open("gabage.txt", "w")  # random outputs
PLOT = True  # If needed to do the final plotting
saveEnergy = True  # If needed to save energy when QoSes can all be satisfied
helpID = 0  # Application ID that is being helped. >0 means upSize, <0 means Downsize
victimID = 0  # Application that is helping helpID, thus is a innocent victim
TOLERANCE = 5  # Check these times of latency whenver an action is taken

# if len(sys.argv) == 6:
#     TIMELIMIT = sys.argv[5]
# TIMELIMIT = "0"

def init():
    global EWAY, MLat, TIMELIMIT, CONFIG, NUM, APP, QoS, Lat, Slack, ECORES, CORES, FREQ, WAY, CPU, MEM, INTERVAL
    # if len(sys.argv) > 2:
    #     TIMELIMIT = int(sys.argv[2])
    # Read the name of colocated applications and their QoS target (may be in different units)
    print("initialize!")
    if os.path.isfile("%s" % CONFIG) == False:
        print("config file (%s) does not exist!" % CONFIG)
        exit(0)
    with open("%s" % CONFIG, "r") as f:
        lines = f.readlines()
        assert len(lines[0].split()) == 1
        NUM = int(lines[0].split()[0])
        assert len(lines) >= (NUM + 1)
        for i in range(1, NUM + 1, 1):
            words = lines[i].split()
            assert len(words) == 2
            CORES[i] = []
            APP[i] = words[0]
            # IP[i] = words[1]
            # assert APP[i] in QOS
            QoS[i] = int(words[1])
            WAY[i] = TOTAL_LLC // NUM
            MLat[i] = collections.deque(maxlen=(int(1.0 / INTERVAL)))
            MEM[i] = int(100 / NUM)
    # Initialize resource parititioning
    j = 0
    while len(ECORES) > 0:
        CORES[j + 1].append(ECORES.pop())
        j = (j + 1) % NUM
    for i in range(TOTAL_LLC - TOTAL_LLC // NUM * NUM):
        WAY[i + 1] += 1

    # Enforce harware isolation
    propogateCore()
    # Monitoring of CPU and cache utilizataion is not needed in PARTIES manager. You can comment them out. These are just legacy codes and may be useful if you want to monitor real-time resource usage.
    # monproc = subprocess.Popen("python /home/sc2682/scripts/monitor/monitorN.py %d" % TIMELIMIT, shell=True, stdout=FF, stderr=FF, preexec_fn=os.setsid);


def main():
    global TIMELIMIT, RESNET_QPS, BERT_QPS
    init()
    print("after initiation...\n")
    sleep(1)
    currentTime = 0

    changeQPS()

    print(RESNET_QPS + " " + BERT_QPS)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "w") as file:
        file.write(f"Start executing at {formatted_datetime}\n")

    flag = True
    while flag:
        # wait()
        flag = makeDecision()
        printState()


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


def printState():
    print("Current state after round: ", ROUND)
    for i in range(1, NUM + 1):
        print(APP[i], ": Cores: ", len(CORES[i]), " LLC: ", WAY[i], " Freq: ", FREQ[i])
    print("\n")
    with open(LOG_FILE, "a") as file:
        for i in range(1, NUM + 1):
            file.write(
                "{}: Cores: {} LLC: {} MB: {}\n".format(
                    APP[i], len(CORES[i]), WAY[i], MEM[i]
                )
            )
        file.write("\n")


def makeDecision():
    global Lat, LSlack, TOLERANCE, LLSlack, REST, Slack, NUM, FREQ, helpID, victimID
    print("Make a decision! ", helpID)
    with open(LOG_FILE, "a") as file:
        file.write(f"Make a decision! {helpID}\n")
    if helpID > 0:
        cur = Lat[helpID]
        cnt = 0
        print("Start executing for ", TOLERANCE, " times")
        with open(LOG_FILE, "a") as file:
            file.write(f"Start executing for {TOLERANCE} times\n")
        for i in range(TOLERANCE):
            wait()
            if Lat[helpID] < cur:
                cnt += 1
            else:
                cnt -= 1
        if cnt <= 4 or (State[helpID] == 2 and FREQ[helpID] == 2300):
            # return revert(helpID)
            revert(helpID)
        # else:
        #     cnt = 0
        #     wait()
        #     while (Lat[helpID] < cur):
        #         cur = Lat[helpID]
        #         wait()
        helpID = victimID = 0
    elif helpID < 0:
        cur = Lat[-helpID]
        cnt = 0
        print("Start executing for ", TOLERANCE, " times")
        with open(LOG_FILE, "w") as file:
            file.write(f"Start executing for {TOLERANCE} times\n")
        for i in range(TOLERANCE):
            wait()
            flag = True
            for j in range(1, NUM + 1):
                if Slack[j] < 0:
                    flag = False
                    break
            if flag == False:
                cnt -= 1
            else:
                cnt += 1
        if cnt <= 4:
            # return revert(-helpID)  # Revert back as it doesn't benefit from this resource
            revert(helpID)
            #  wait()
            cnt = 0
            print("Start executing for ", TOLERANCE, " times")
            with open(LOG_FILE, "a") as file:
                file.write(f"Start executing for {TOLERANCE} times\n")
            for i in range(TOLERANCE):
                wait()
                flag = True
                for j in range(1, NUM + 1):
                    # if Slack[j] < 0.05:
                    if Slack[j] < 0:
                        flag = False
                        break
                if flag == False:
                    cnt -= 1
                else:
                    cnt += 1
            if cnt <= 4:
                print("Start executing for ", TOLERANCE, " times")
                with open(LOG_FILE, "a") as file:
                    file.write(f"Start executing for {TOLERANCE} times\n")
                for i in range(TOLERANCE):
                    wait()
        #  while Slack[-helpID] < 0 or LSlack[-helpID] < 0:
        #      print "wait..."
        #      wait()
        helpID = victimID = 0
    else:
        print("Start executing for 1 times")
        with open(LOG_FILE, "a") as file:
            file.write("Start executing for 1 times" + "\n")
        wait()
    if helpID == 0:  # Don't need to check any application before making a new decision
        idx = -1
        victimID = 0
        for i in range(1, NUM + 1):  # First find if any app violates QoS
            # if Slack[i] < 0.05 and LSlack[i] < 0.05:
            if Slack[i] < 0 and LSlack[i] < 0:
                if idx == -1 or LSlack[i] < LSlack[idx]:
                    idx = i
            elif (
                (LDOWN[i] == 0)
                and Slack[i] > 0.2
                and LSlack[i] > 0.2
                and (victimID == 0 or Slack[i] > Slack[victimID])
            ):
                victimID = i
        if idx != -1:
            return upSize(idx)  # If found, give more resources to this app
        elif (
            saveEnergy == True and victimID > 0
        ):  # If not found, try removing resources
            return downSize(victimID)
        else:
            # print("Start executing for 1 times")
            # with open(LOG_FILE, "a") as file:
            #     file.write("Start executing for 1 times" + "\n")
            # wait()
            print(
                "All QoS are met, and slack for each process is not big enough to do downsizing. Program exiting.\nCurrent slack is 0.2.\n"
            )
            with open(LOG_FILE, "a") as file:
                file.write(
                    "All QoS are met, and slack for each process is not big enough to do downsizing. Program exiting.\nCurrent slack is 0.2.\n"
                )
            return False
    return True

def nextState(idx, upsize=True):
    global State
    if State[idx] == 0:
        if upsize == True:
            State[idx] = random.choice([1, 2, 3])
        else:
            State[idx] = -random.choice([1, 2, 3])
    elif State[idx] == -1:
        State[idx] = -3
    elif State[idx] == -2:
        State[idx] = -1
    elif State[idx] == -3:
        State[idx] = -2
    elif State[idx] == 1:
        State[idx] = 3
    elif State[idx] == 2:
        State[idx] = 1
    elif State[idx] == 3:
        State[idx] = 2
    else:
        assert False


def revert(idx):
    global State, APP, helpID, victimID, REST
    print(idx, " revert back")
    with open(LOG_FILE, "a") as file:
        file.write(f"{idx} revert back\n")
    if idx < 0:
        if State[-idx] == -1:
            assert adjustCore(-idx, 1, False) == True
        elif State[-idx] == -2:
            assert adjustMemoryBandwidth(-idx, 10) == True
        elif State[-idx] == -3:
            assert adjustCache(-idx, 1, False) == True
        else:
            assert False
        nextState(-idx)
        LDOWN[-idx] = REST
    else:
        nextState(idx)
    return True


def upSize(idx):
    global State, helpID, victimID, APP
    victimID = 0
    helpID = idx
    print("Upsize ", APP[idx], "(", State[idx], ")")
    with open(LOG_FILE, "a") as file:
        file.write(f"Upsize {APP[idx]} ({State[idx]})\n")

    if State[idx] <= 0:
        State[idx] = random.choice([1, 2, 3])
    for k in range(3):
        if (
            (State[idx] == 1 and adjustCore(idx, 1, False) == False)
            or (State[idx] == 2 and adjustMemoryBandwidth(idx, 10) == False)
            or (State[idx] == 3 and adjustCache(idx, 1, False) == False)
        ):
            nextState(idx)
        else:
            return True
    print("No way to upsize any more...")
    with open(LOG_FILE, "a") as file:
        file.write("No way to upsize any more..." + ")\n")
    helpID = 0
    return False


def downSize(idx):
    global State, helpID, victimID
    print("Downsize ", APP[idx], "(", State[idx], ")")
    with open(LOG_FILE, "a") as file:
        file.write(f"Downsize {APP[idx]}({State[idx]})\n")
    victimID = 0
    if State[idx] >= 0:
        State[idx] = -random.randint(1, 3)
    for k in range(3):
        if (
            (State[idx] == -1 and adjustCore(idx, -1, False) == False)
            or (State[idx] == -2 and adjustMemoryBandwidth(idx, -10) == False)
            or (State[idx] == -3 and adjustCache(idx, -1, False) == False)
        ):
            nextState(idx)
        else:
            helpID = -idx
            return True
    print("No way to downsize any more...")
    with open(LOG_FILE, "a") as file:
        file.write("No way to downsize any more..." + ")\n")
    return False

def getLat():
    global APP, Lat, MLat, LLSlack, LSlack, Slack, QoS, NUM
    # print(LATENCY_FILE, flush=True)
    # with open(LOG_FILE, "a") as file:
    #     file.write(LATENCY_FILE)
    with open(LATENCY_FILE, "r") as file:
        # Skip the first and second line (caption)
        file.readline()
        file.readline()

        # Extract values for the given number of lines
        all_violation = 0
        for i in range(1, NUM + 1):
            app = APP[i]
            if APP[i][-1] == "2":
                app = APP[i][:-1]

            line = file.readline().strip()
            values = line.split()
            LLSlack[i] = Slack[i]
            # Convert to float, multiply by 10^6, and store in the dictionary
            Lat[i] = float(values[13]) * 10**6
            MLat[i].append(float(values[13]) * 10**6)
            LSlack[i] = 1 - sum(MLat[i]) * 1.0 / len(MLat[i]) / QoS[i]

            Slack[i] = (QoS[i] - Lat[i]) * 1.0 / QoS[i]
            if Slack[i] < ALL_VIOLATION_LIMIT:
                all_violation += 1
            # print("  --", APP[i], ":", Lat[i], "(", Slack[i], LSlack[i], ")", flush=True)
            with open(LOG_FILE, "a") as record:
                record.write(f"{APP[i]}: {float(values[13])}\n")
        if all_violation == NUM:
            exit(0)

def wait():
    global INTERVAL, TIMELIMIT, ROUND
    # sleep(INTERVAL)
    # # Run code here
    # print("Round: ", ROUND, flush=True)
    # with open(LOG_FILE, "a") as file:
    #     file.write("Round: ")
    subprocess.run(["bash", RUN_SCRIPT])
    # print("Flag", flush=True)
    ROUND += 1
    for i in range(1, NUM + 1):
        if LDOWN[i] > 0:
            LDOWN[i] -= 1
    getLat()
    # getData()
    # record()
    if TIMELIMIT != -1:
        TIMELIMIT -= INTERVAL
        if TIMELIMIT < 0:
            # printout()
            exit(0)

def coreStr(cores):
    return ",".join(str(e) for e in cores)


def coreStrHyper(cores):
    return coreStr(cores) + "," + ",".join(str(e + 44) for e in cores)


def way(ways, rightways):
    return hex(int("1" * ways + "0" * rightways, 2))


def adjustCore(idx, num, hasVictim):
    global State, CORES, Slack, ECORES, victimID
    if num < 0:
        if len(CORES[idx]) <= -num:
            return False
        if hasVictim == False or victimID == 0:
            for i in range(-num):
                ECORES.append(CORES[idx].pop())
        else:
            for i in range(-num):
                CORES[victimID].append(CORES[idx].pop())
            # propogateCore(victimID)
    else:
        assert num == 1 and hasVictim == False
        if len(ECORES) >= 1:
            CORES[idx].append(ECORES.pop())
        else:
            victimID = 0
            for i in range(1, NUM + 1):
                if (
                    i != idx
                    and len(CORES[i]) > 1
                    and (victimID == 0 or Slack[i] > Slack[victimID])
                ):
                    victimID = i
            if victimID == 0:
                return False
            CORES[idx].append(CORES[victimID].pop())
            if State[idx] == State[victimID]:
                nextState(victimID)
            # propogateCore(victimID)
    propogateCore()
    print("Adjust core")
    with open(LOG_FILE, "a") as file:
        file.write("Adjust core" + "\n")
    return True


def adjustCache(idx, num, hasVictim):
    global WAY, EWAY, NUM, victimID, State, Slack
    if num < 0:
        if WAY[idx] <= -num:
            return False
        if hasVictim == False or victimID == 0:
            EWAY -= num
        else:
            WAY[victimID] -= num
            # propogateCache(victimID)
    else:
        assert num == 1 and hasVictim == False
        if EWAY > 0:
            EWAY -= 1
        else:
            victimID = 0
            for i in range(1, NUM + 1):
                if (
                    i != idx
                    and WAY[i] > 1
                    and (victimID == 0 or Slack[i] > Slack[victimID])
                ):
                    victimID = i
            if victimID == 0:
                return False
            WAY[victimID] -= num
            # propogateCache(victimID)
            if State[idx] == State[victimID]:
                nextState(victimID)
    WAY[idx] += num
    propogateCache()
    print("Adjust cache")
    with open(LOG_FILE, "a") as file:
        file.write("Adjust cache" + "\n")
    return True


def adjustMemoryBandwidth(idx, num):
    global MEM, APP, State

    assert MEM[idx] > 0 and MEM[idx] < 100
    if num < 0:
        if MEM[idx] <= -num:
            return False  # Memory bandwidth is already at the lowest. Cannot be reduced further
        else:
            MEM[idx] += num
            MEM[3 - idx] = 100 - MEM[idx]
            propogateMemoryBandwidth(idx)
    else:
        if MEM[idx] >= 100 - num:
            return False
        else:
            MEM[idx] += num
            MEM[3 - idx] = 100 - MEM[idx]
            propogateMemoryBandwidth(idx)
    print("Adjust memory bandwidth")
    with open(LOG_FILE, "a") as file:
        file.write("Adjust memory bandwidth" + "\n")
    return True


def propogateCore(idx=None):
    # print("Propogate core")
    global APP, CORES, NUM

    with open(RUN_SCRIPT, "r") as file:
        lines = file.readlines()

    # Current core end
    current_end = -1

    # Modify the script using CORES
    for i in range(1, NUM + 1):
        app_name = APP[i]
        cores_range = CORES[i]

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

    propogateCache()
    propogateMemoryBandwidth()


def propogateCache(idx=None):
    # print("Propogate cache")
    global CORES, WAY, NUM, APP

    def ways_to_llc(ways_value, start_position, length=12):
        """Convert the way count to a binary representation, then to its hexadecimal form."""
        # Initialize an empty binary string
        llc_bin = ["0"] * length
        # Set the specific positions to 1 based on the ways_value
        for i in range(ways_value):
            print(start_position - i)
            llc_bin[start_position - i] = "1"
        # Convert the list of binary digits to a string and then to its hexadecimal format
        hex_value = hex(int("".join(llc_bin), 2))
        pad = ((length-1-1) // 4) + 1
        return "0x" + hex_value[2:].rjust(
            pad, "0"
        )  # Pad with leading zeros to ensure 3 characters

    with open(RUN_SCRIPT, "r") as file:
        lines = file.readlines()

    current_position = TOTAL_LLC  # Start from the rightmost position

    for i in range(1, NUM + 1):
        app_name = APP[i]

        # Update script based on the WAYS values
        ways_value = WAY[i]
        llc_hex = ways_to_llc(ways_value, current_position, TOTAL_LLC+1)
        current_position -= ways_value  # update the starting position for the next APP

        for idx, line in enumerate(lines):
            if line.startswith(app_name + "_llc"):
                lines[idx] = f"{app_name}_llc={llc_hex}\n"

    # Write the modified content back to the file
    with open(RUN_SCRIPT, "w") as file:
        file.writelines(lines)


def propogateMemoryBandwidth(idx=None):
    global MEM, NUM, APP

    with open(RUN_SCRIPT, "r") as file:
        lines = file.readlines()

    for i in range(1, NUM + 1):
        app_name = APP[i]
        MEM_value = MEM[i]

        for idx, line in enumerate(lines):
            if line.startswith(app_name + "_mb"):
                lines[idx] = f"{app_name}_mb={MEM_value}\n"

    # Write the modified content back to the file
    with open(RUN_SCRIPT, "w") as file:
        file.writelines(lines)

if __name__ == "__main__":
    main()
