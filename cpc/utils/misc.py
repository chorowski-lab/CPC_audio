# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import numpy as np
import random
import torch
import sys
import psutil
from copy import deepcopy
from bisect import bisect_left


def untensor(d):
    if isinstance(d, list):
        return [untensor(v) for v in d]
    if isinstance(d, dict):
        return dict((k, untensor(v)) for k, v in d.items())
    if hasattr(d, 'tolist'):
        return d.tolist()
    return d


def save_logs(data, pathLogs):
    with open(pathLogs, 'w') as file:
        json.dump(data, file, indent=2)


def update_logs(logs, logStep, prevlogs=None):
    out = {}
    for key in logs:
        out[key] = deepcopy(logs[key])

        if prevlogs is not None:
            out[key] -= prevlogs[key]
        out[key] /= logStep
    return out


def show_logs(text, logs):
    print("")
    print('-'*50)
    print(text)

    for key in logs:

        if key == "iter":
            continue

        if key.startswith("grad"):
            print(f"{key}: {logs[key]}")
            continue

        if key == "pushloss_closest":
            print("closest to protos:", ", ".join([str(x) for x in logs[key]]))
            continue

        if key == "labelCounts":

            cnt = torch.tensor(logs[key])
            if cnt.shape[0] < 5 or cnt.shape[1] < 5:  # empty no-stats-yet data
                continue
            #print(cnt.shape,  cnt.sum(dim=1).view(-1,1).shape, cnt.sum(dim=0).view(1,-1).shape)
            topForClusters = torch.topk(cnt / torch.clamp(cnt.sum(dim=1).view(-1,1), min=1), 5, dim=1)
            topForPhones = torch.topk(cnt / torch.clamp(cnt.sum(dim=0).view(1,-1), min=1), 5, dim=0)
            topForPhonesValues = topForPhones.values.transpose(0,1)
            topForPhonesIndices = topForPhones.indices.transpose(0,1)
            #print("::::::", cnt.shape, topForClusters.values.shape, topForPhones.values.shape)

            topForClustersSums = torch.zeros(3, dtype=float)
            topForClustersCounts = 0
            topForClustersSumsNoPh0 = torch.zeros(3, dtype=float)
            topForClustersCountsNoPh0 = 0
            topForPhonesSums = torch.zeros(3, dtype=float)
            topForPhonesCounts = 0
            print("-----------> top occ for clusters in 0-1 format:")
            for i in range(topForClusters.indices.shape[0]):
                print(str(i), ":|", ", ".join(map(lambda a: str(a[0].item())+": "+str("{:.4f}".format(a[1].item())), zip(topForClusters.indices[i], topForClusters.values[i]))))
                if topForClusters.values[i][0] > 0.000001:  # non-zeroed cluster
                    topForClustersCounts += 1
                    for to in range(3):
                        for where in range(to + 1):
                            topForClustersSums[to] += topForClusters.values[i][where]
                    if topForClusters.indices[i][0] != 0:
                        topForClustersCountsNoPh0 += 1
                        for to in range(3):
                            for where in range(to + 1):
                                topForClustersSumsNoPh0[to] += topForClusters.values[i][where]
            topForClustersSums /= topForClustersCounts
            topForClustersSumsNoPh0 /= topForClustersCountsNoPh0
            print(f"averages of top 1-3 sums for non-zeroed ({topForClustersCounts} clusters): top1 {topForClustersSums[0]}, top2 {topForClustersSums[1]}, top3 {topForClustersSums[2]}")
            print(f"averages of top 1-3 sums for non-zeroed non-ph-0 ({topForClustersCountsNoPh0} clusters): top1 {topForClustersSumsNoPh0[0]}, top2 {topForClustersSumsNoPh0[1]}, top3 {topForClustersSumsNoPh0[2]}")
            print("-----------> top occ for phonemes in 0-1 format:")
            for i in range(topForPhonesIndices.shape[0]):
                print(str(i), ":|", ", ".join(map(lambda a: str(a[0].item())+": "+str("{:.4f}".format(a[1].item())), zip(topForPhonesIndices[i], topForPhonesValues[i]))))
                if topForPhonesValues[i][0] > 0.000001:  # non-zeroed cluster
                    topForPhonesCounts += 1
                    for to in range(3):
                        for where in range(to + 1):
                            topForPhonesSums[to] += topForPhonesValues[i][where]
            topForPhonesSums /= topForPhonesCounts
            print(f"averages of top 1-3 sums for non-zeroed ({topForPhonesCounts}): top1 {topForPhonesSums[0]}, top2 {topForPhonesSums[1]}, top3 {topForPhonesSums[2]}")
                

            continue

        if key == "centersDM":

            DM = logs[key]
            if DM.shape[0] < 5 or DM.shape[1] < 5:  # empty no-stats-yet data
                continue
            print(f"--------> DM avg distances, mean: {DM.mean()}")
            print(", ".join(map(lambda a: f"({a[0]}-{a[1]}): {a[2]}", [(i, j, "{:.3e}".format(DM[i,j])) for i in range(DM.shape[0]) for j in range(i, DM.shape[1])])))
            print("---")

            continue

        nPredicts = logs[key].shape[0]

        strSteps = ['Step'] + [str(s) for s in range(1, nPredicts + 1)]
        formatCommand = ' '.join(['{:>16}' for x in range(nPredicts + 1)])
        print(formatCommand.format(*strSteps))

        strLog = [key] + ["{:10.6f}".format(s) for s in logs[key]]
        print(formatCommand.format(*strLog))

    print('-'*50)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())


def ramp_scheduling_function(n_epoch_ramp, epoch):
    if epoch >= n_epoch_ramp:
        return 1
    else:
        return (epoch + 1) / n_epoch_ramp


class SchedulerCombiner:
    r"""
    An object which applies a list of learning rate schedulers sequentially.
    """

    def __init__(self, scheduler_list, activation_step, curr_step=0):
        r"""
        Args:
            - scheduler_list (list): a list of learning rate schedulers
            - activation_step (list): a list of int. activation_step[i]
            indicates at which step scheduler_list[i] should be activated
            - curr_step (int): the starting step. Must be lower than
            activation_step[0]
        """

        if len(scheduler_list) != len(activation_step):
            raise ValueError("The number of scheduler must be the same as "
                             "the number of activation step")
        if activation_step[0] > curr_step:
            raise ValueError("The first activation step cannot be higher than "
                             "the current step.")
        self.scheduler_list = scheduler_list
        self.activation_step = deepcopy(activation_step)
        self.curr_step = curr_step

    def step(self):
        self.curr_step += 1
        index = bisect_left(self.activation_step, self.curr_step) - 1
        for i in reversed(range(index, len(self.scheduler_list))):
            self.scheduler_list[i].step()

    def __str__(self):
        out = "SchedulerCombiner \n"
        out += "(\n"
        for index, scheduler in enumerate(self.scheduler_list):
            out += f"({index}) {scheduler.__str__()} \n"
        out += ")\n"
        return out
