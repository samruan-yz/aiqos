#!/usr/bin/python
from time import sleep
import random
import sys
import os
import collections
import threading
import argparse
from datetime import datetime

# sys.path.append('/home/sc2682/scripts/monitor')
# from monitorN import startMonitoring, endMonitoring
import subprocess


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process Model latency and file paths')
    # adding new arguments
    parser.add_argument('--RESNET_QPS', type=int, required=True, help='RESNET QPS')
    parser.add_argument('--BERT_QPS', type=int, required=True, help='BERT QPS')
    parser.add_argument('--CONFIG', type=str, default="/data3/ydil/aiqos/inference_results_v1.1/closed/Intel/code/input.txt", help='PATH of config file')
    parser.add_argument('--LATENCY_FILE', type=str, default=None, help='Path to the latency file')
    parser.add_argument('--RUN_SCRIPT', type=str, default="/data3/ydil/aiqos/inference_results_v1.1/closed/Intel/code/collocation_run_resnet_bert.sh", help='Path to the run script')
    parser.add_argument('--LOG_FILE', type=str, default=None, help='Path to the log file')
    
    args = parser.parse_args()
    
    if args.LATENCY_FILE is None:
        args.LATENCY_FILE =  f"/data3/ydil/aiqos/inference_results_v1.1/closed/Intel/code/data_point/resnet_bert_search/result_{args.RESNET_QPS}_{args.BERT_QPS}.data"
    if args.LOG_FILE is None:
        args.LOG_FILE = f"/data3/ydil/aiqos/inference_results_v1.1/closed/Intel/code/PARTIES_LOG/parties_res_{args.RESNET_QPS}_{args.BERT_QPS}.data"
        
    return args
# if len(sys.argv) > 1:
#     CONFIG = sys.argv[1]

# QoS target of each application, in nanoseconds.
# QOS = {
#     "moses": 15000000,
#     "xapian": 5000000,
#     "nginx": 10000000,
#     "sphinx": 2500000000,
#     "memcached": 600000,
#     "mongodb": 300000000,
# }

ROUND = 0
INTERVAL = 0.1  # Frequency of monitoring, unit is second
TIMELIMIT = 60  # How long to run this controller, unit is in second.
REST = 100
NUM = 0  # Number of colocated applications
APP = [None for i in range(10)]  # Application names
IP = [None for i in range(10)]  # IP of clients that run applications
QoS = [None for i in range(10)]  # Target QoS of each application
ECORES = [i for i in range(0, 24, 1)]  # unallocated cores
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

ITERATIONS = ["6", "4", "1", "5","7", "24", "17"]
CONFIGURATIONS = ["R:12,9,60% B:12,2,40%", "R:12,8,50% B:12,3,50%" ,"R:12,6,40% B:12,5,60%", "R:11,3,40% B:13,8,60","R:8,4,40% B:16,7,60%", "R:7,4,10% B:17,7,90%", "R:7,1,10% B:17,10,90%"]


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
            WAY[i] = 11 // NUM
            MLat[i] = collections.deque(maxlen=(int(1.0 / INTERVAL)))
            MEM[i] = int(100 / NUM)
    # Initialize resource parititioning
    j = 0
    while len(ECORES) > 0:
        CORES[j + 1].append(ECORES.pop())
        j = (j + 1) % NUM
    for i in range(11 - 11 // NUM * NUM):
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
        if cnt <= 0 or (State[helpID] == 2 and FREQ[helpID] == 2300):
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
                if Slack[j] < 0.05:
                    flag = False
                    break
            if flag == False:
                cnt -= 1
            else:
                cnt += 1
        if cnt <= 0:
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
                    if Slack[j] < 0.05:
                        flag = False
                        break
                if flag == False:
                    cnt -= 1
                else:
                    cnt += 1
            if cnt <= 0:
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
            if Slack[i] < 0.05 and LSlack[i] < 0.05:
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


# def makeDecision():
#     global Lat, LSlack, TOLERANCE, LLSlack, REST, Slack, NUM, FREQ, helpID, victimID
#     print("Make a decision! ", helpID)
#     with open('decision.txt', 'a') as file:
#         file.write(f"Make a decision! {helpID}\n")
#     if helpID > 0:
#         cur = Lat[helpID]
#         latencies = []
#         print("Start executing for ", TOLERANCE, " times")
#         with open('decision.txt', 'a') as file:
#             file.write(f"Start executing for {TOLERANCE} times\n")
#          print("Start executing for ", TOLERANCE, " times")
#         with open('decision.txt', 'w') as file:
#             file.write(f"Start executing for {TOLERANCE} times\n")
#         for i in range(TOLERANCE):
#             wait()
#             slacks.append(Slack[-helpID])
#         if sum(slacks) / len(slacks) < 0.05:
#             revert(helpID)
#         helpID = victimID = 0
#     else:
#         print("Start executing for 1 times")
#         with open('decision.txt', 'a') as file:
#                 file.write("Start executing for 1 times" + "\n")
#         wait()
#     if helpID == 0:  # Don't need to check any application before making a new decision
#         idx = -1
#         victimID = 0
#         for i in range(1, NUM + 1):  # First find if any app violates QoS
#             if Slack[i] < 0.05 and LSlack[i] < 0.05:
#                 if idx == -1 or LSlack[i] < LSlack[idx]:
#                     idx = i
#             elif (
#                 (LDOWN[i] == 0)
#                 and Slack[i] > 0.2
#                 and LSlack[i] > 0.2
#                 and (victimID == 0 or Slack[i] > Slack[victimID])
#             ):
#                 victimID = i
#         if idx != -1:
#             return upSize(idx)  # If found, give more resources to this app
#         elif (
#             saveEnergy == True and victimID > 0
#         ):  # If not found, try removing resources
#             return downSize(victimID)
#         else:
#             print("Start executing for 1 times")
#             with open('decision.txt', 'a') as file:
#                     file.write("Start executing for 1 times" + "\n")
#             wait()
#     return True
#   for i in range(TOLERANCE):
#             wait()
#             latencies.append(Lat[helpID])
#         if sum(latencies) / len(latencies) >= cur or (State[helpID] == 2 and FREQ[helpID] == 2300):
#             revert(helpID)
#         helpID = victimID = 0
#     elif helpID < 0:
#         cur = Lat[-helpID]
#         slacks = []

# FSM state of resource adjustment
# -3: give it fewer cache
# -2: give it fewer frequency
# -1: give it fewer cores
#  0: not in adjustment
#  1: give it more cores
#  2: give it more frequency
#  3: give it more cache


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


def wait():
    global INTERVAL, TIMELIMIT, ROUND
    # sleep(INTERVAL)
    # # Run code here
    # print("Round: ", ROUND)
    subprocess.call(["bash", RUN_SCRIPT])
    ROUND += 1
    for i in range(1, NUM + 1):
        if LDOWN[i] > 0:
            LDOWN[i] -= 1
    getLat()
    getData()
    record()
    if TIMELIMIT != -1:
        TIMELIMIT -= INTERVAL
        if TIMELIMIT < 0:
            # printout()
            exit(0)


def getLat():
    global APP, Lat, MLat, LLSlack, LSlack, Slack, QoS, NUM
    with open(LATENCY_FILE, "r") as file:
        # Skip the first and second line (caption)
        file.readline()
        file.readline()

        # Extract values for the given number of lines
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
            # print("  --", APP[i], ":", Lat[i], "(", Slack[i], LSlack[i], ")")]
            with open(LOG_FILE, "a") as record:
                record.write(f"{APP[i]}: {float(values[13])}\n")


def getData():
    global NUM, cCPU, CPU, CORES, MEM
    tmp = 0
    # Monitoring of CPU and cache utilizataion is not needed in PARTIES manager. You can comment them out. These are just legacy codes and may be useful if you want to monitor real-time resource usage.
    # with open("/home/sc2682/scripts/monitor/cpu.txt", "r") as ff:
    #    lines = ff.readlines();
    #    while (len(lines) >=1 and "Average" in lines[-1]):
    #        lines = lines[:-1]
    #    if (len(lines) >= 22):
    #        lines = lines[-22:]
    #        cnt = [0 for i in xrange(0, NUM+10, 1)]
    #        for line in lines:
    #            if "Average" in line:
    #                break
    #            words = line.split()
    #            if len(words)<10:
    #                break
    #            cpuid = int(words[2])
    #            tmp += float(words[3])+float(words[5])+float(words[6])+float(words[8])
    #            for j in xrange(1, NUM+1, 1):
    #                if cpuid in CORES[j]:
    #                    CPU[j] += float(words[3])+float(words[5])+float(words[6])+float(words[8])
    #                    cnt[j] += 1
    #                break
    #        for j in xrange(1, NUM+1):
    #            if cnt[j] > 0:
    #                CPU[j] /= cnt[j]
    # cCPU.append(tmp/14.0)

    # with open("/home/sc2682/scripts/monitor/cat.txt", "r") as ff:
    #    lines = ff.readlines();
    #    if (len(lines) >= 22):
    #        lines = lines[-22:]
    #        for line in lines:
    #            words = line.split()
    #            if words[0] == "TIME" or words[0] == "CORE" or words[0] == "WARN":
    #                continue
    #            if ("WARN:" in words[0]) or ("Ter" in words[0]):
    #                break
    #            cpuid = int(words[0])
    #            for j in xrange(1, NUM+1):
    #                if cpuid in CORES[j]:
    #                    MEM[j] += float(words[4])+float(words[5])


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
            llc_bin[start_position - i] = "1"
        # Convert the list of binary digits to a string and then to its hexadecimal format
        hex_value = hex(int("".join(llc_bin), 2))
        return "0x" + hex_value[2:].rjust(
            3, "0"
        )  # Pad with leading zeros to ensure 3 characters

    with open(RUN_SCRIPT, "r") as file:
        lines = file.readlines()

    current_position = 11  # Start from the rightmost position

    for i in range(1, NUM + 1):
        app_name = APP[i]

        # Update script based on the WAYS values
        ways_value = WAY[i]
        llc_hex = ways_to_llc(ways_value, current_position)
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


def record():
    global CPU, LOAD, NUM, Lat, rrLat, saveEnergy, rLat, CORES, rCORES, WAY, rWAY, FREQ, rFREQ
    # for i in range(1, NUM + 1):
    #     rrLat[i].append(Lat[i])
    #     rLat[i].append(1 - LSlack[i])
    #     rCORES[i].append(len(CORES[i]))
    #     rWAY[i].append(WAY[i])
    #     rFREQ[i].append(FREQ[i])
    # p = subprocess.Popen(
    #     "curl http://128.253.128.66:84/memcached/count.txt | tail -1",
    #     shell=True,
    #     stdout=subprocess.PIPE,
    #     stderr=FF,
    #     preexec_fn=os.setsid,
    #     bufsize=0,
    # )
    # # LOAD.append(10)
    # out, err = p.communicate()
    # if out != "":
    #     LOAD.append(int(out) - 1)
    #     # if int(out) > 24:
    #     #    saveEnergy = False
    #     # else:
    #     #    saveEnergy = True
    # elif len(LOAD) > 0:
    #     LOAD.append(LOAD[-1])
    # else:
    #     LOAD.append(0)


def printout():
    global NUM, rrLat, LOAD, rLat, rCORES, cCPU, rFREQ, rWAY
    print("CPU Utilization: ", sum(cCPU) * 1.0 / len(cCPU))
    if PLOT == True:
        of = open("/home/sc2682/scripts/manage/results.txt", "w")
        for i in range(1, NUM + 1):
            for item in rrLat[i]:
                of.write("%d " % item)
            of.write("\n")
            for item in rLat[i]:
                of.write("%f " % item)
            of.write("\n")
            for item in rCORES[i]:
                of.write("%d " % item)
            of.write("\n")
            for item in rFREQ[i]:
                of.write("%d " % item)
            of.write("\n")
            for item in rWAY[i]:
                of.write("%d " % item)
            of.write("\n")
        for item in LOAD:
            of.write("%d " % item)
        of.write("\n")
        of.close()


def adjustFreq(idx, num):
    # global FREQ, APP, State
    # assert FREQ[idx] >= 1200 and FREQ[idx] <= 2300
    # if num < 0:
    #     if FREQ[idx] == 1200:
    #         return (
    #             False  # Frequency is already at the lowest. Cannot be reduced further
    #         )
    #     else:
    #         FREQ[idx] += 100 * num
    #         propogateFreq(idx)
    # else:
    #     if FREQ[idx] == 2300:
    #         return False  # Shuang
    #         victimID = 0
    #         for i in range(1, NUM + 1):
    #             if (
    #                 i != idx
    #                 and FREQ[i] > 1200
    #                 and (victimID == 0 or Slack[i] > Slack[victimID])
    #             ):
    #                 victimID = i
    #         if victimID == 0:
    #             return False
    #         else:
    #             FREQ[victimID] -= 100 * num
    #             propogateFreq(victimID)
    #             if State[victimID] == State[idx]:
    #                 nextState(victimID)
    #     else:
    #         FREQ[idx] += 100 * num
    #         propogateFreq(idx)
    print("Adjust freq")
    return True


def propogateFreq(idx=None):
    # print("Propogate freq")
    global CORES, FREQ, NUM, APP
    # if idx == None:
    #     subprocess.call(
    #         ["cpupower", "-c", "0-87", "frequency-set", "-g", "userspace"],
    #         stdout=FF,
    #         stderr=FF,
    #     )
    #     subprocess.call(
    #         ["cpupower", "-c", "0-87", "frequency-set", "-f", "2200MHz"],
    #         stdout=FF,
    #         stderr=FF,
    #     )
    #     for i in range(1, NUM + 1):
    #         print("    Change Frequency of", APP[i], ":", FREQ[i])
    #         if FREQ[i] <= 2200:
    #             subprocess.call(
    #                 [
    #                     "cpupower",
    #                     "-c",
    #                     coreStrHyper(CORES[i]),
    #                     "frequency-set",
    #                     "-f",
    #                     "%dMHz" % FREQ[i],
    #                 ],
    #                 stdout=FF,
    #                 stderr=FF,
    #             )
    #         else:
    #             subprocess.call(
    #                 [
    #                     "cpupower",
    #                     "-c",
    #                     coreStrHyper(CORES[i]),
    #                     "frequency-set",
    #                     "-g",
    #                     "performance",
    #                 ],
    #                 stdout=FF,
    #                 stderr=FF,
    #             )
    # else:
    #     print("    Change Frequency of", APP[idx], ":", FREQ[idx])
    #     if FREQ[idx] <= 2200:
    #         subprocess.call(
    #             [
    #                 "cpupower",
    #                 "-c",
    #                 coreStrHyper(CORES[idx]),
    #                 "frequency-set",
    #                 "-g",
    #                 "userspace",
    #             ],
    #             stdout=FF,
    #             stderr=FF,
    #         )
    #         subprocess.call(
    #             [
    #                 "cpupower",
    #                 "-c",
    #                 coreStrHyper(CORES[idx]),
    #                 "frequency-set",
    #                 "-f",
    #                 "%dMHz" % FREQ[idx],
    #             ],
    #             stdout=FF,
    #             stderr=FF,
    #         )
    #     else:
    #         subprocess.call(
    #             [
    #                 "cpupower",
    #                 "-c",
    #                 coreStrHyper(CORES[idx]),
    #                 "frequency-set",
    #                 "-g",
    #                 "performance",
    #             ],
    #             stdout=FF,
    #             stderr=FF,
    #         )


if __name__ == "__main__":
    args = parse_arguments()
    RESNET_QPS = args.RESNET_QPS
    BERT_QPS = args.BERT_QPS
    LATENCY_FILE = args.LATENCY_FILE
    RUN_SCRIPT = args.RUN_SCRIPT
    LOG_FILE = args.LOG_FILE
    main()
