# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import progressbar
import torch
import torch.nn as nn
import cpc.feature_loader as fl
from .. import CTCPhoneCriterion
from os.path import join, exists
from os import remove
from time import time


class kMeanCluster(nn.Module):

    def __init__(self, Ck, norm_vec_len=False):

        super(kMeanCluster, self).__init__()
        self.register_buffer('Ck', Ck)
        self.k = Ck.size(1)
        self.norm_vec_len = norm_vec_len
        print("-----> kMeanCluster init")

    def forward(self, features):
        B, S, D = features.size()
        if self.norm_vec_len:
            #print("kMeanCluster", features.shape)
            featuresLengths = torch.sqrt((features*features).sum(2))
            #print("kMeanCluster", featuresLengths.shape)
            features = features / featuresLengths.view(*(featuresLengths.shape), 1)
        Ck = self.Ck
        
        if self.norm_vec_len:
            #print("centers", Ck.shape)
            CkLengths = torch.sqrt((Ck*Ck).sum(2))
            #print("centers", CkLengths.shape)
            Ck = Ck / CkLengths.view(*(CkLengths.shape), 1)
            clen = torch.sqrt((Ck*Ck).sum(2))
            #print("center lengths after norm min, max", clen.min().item(), clen.max().item())
        features = features.contiguous().view(B*S, 1, -1)
        return ((features - Ck)**2).sum(dim=2).view(-1, S, self.k)


class kMeanClusterStep(torch.nn.Module):

    def __init__(self, k, D, norm_vec_len=False):

        super(kMeanClusterStep, self).__init__()
        self.k = k
        self.register_buffer('Ck', torch.zeros(1, k, D))
        self.norm_vec_len = norm_vec_len

    def forward(self, locF):

        if self.norm_vec_len:
            # print("step", locF.shape)
            locFLengths = torch.sqrt((locF*locF).sum(2))
            # print("step", locFLengths.shape)
            locF = locF / locFLengths.view(*(locFLengths.shape), 1)
            # locFLengths2 = torch.sqrt((locF*locF).sum(2))
            # print("step lengths after norm min, max", locFLengths2.min().item(), locFLengths2.max().item())
        # ckl = torch.sqrt((self.Ck*self.Ck).sum(2))
        # print("cluster lengths after norm min, max", ckl.min().item(), ckl.max().item())
        index = ((locF - self.Ck)**2).mean(dim=2).min(dim=1)[1]
        Ck1 = torch.cat([locF[index == p].sum(dim=0, keepdim=True)
                         for p in range(self.k)], dim=1)
        # locFLengths2 = torch.sqrt((locF*locF).sum(2))
        # print("step lengths 2 after norm min, max", locFLengths2.min().item(), locFLengths2.max().item())
        nItems = torch.cat([(index == p).sum(dim=0, keepdim=True)
                            for p in range(self.k)], dim=0).view(1, -1)
        # print(Ck1.shape, nItems, (Ck1 / nItems.view(1,-1,1)).shape)
        # Ck1NormLengths = ((Ck1 / nItems.view(1,-1,1)) * (Ck1 / nItems.view(1,-1,1))).sum(-1)
        # print("lengths of update:", Ck1NormLengths.shape, Ck1NormLengths.min().item(), Ck1NormLengths.max().item())
        return Ck1, nItems


def kMeanGPU(dataLoader, featureMaker, k, n_group=1,
             MAX_ITER=100, EPSILON=1e-4,
             perIterSize=-1, start_clusters=None,
             save=False, load=False, save_dir=None,
             save_last=5, norm_vec_len=False):

    print(f"Start Kmean clustering with {k} clusters and {n_group} groups...")

    if save or load:
        assert save_dir is not None

    if start_clusters is None:
        if load and exists(join(save_dir, "checkpoint_last.pt")):
            print("Loading from last checkpoint")
            state_dict = torch.load(join(save_dir, "checkpoint_last.pt"))
            Ck = state_dict["state_dict"]["Ck"]
            D = Ck.size(2)
        else:
            Ck = []
            with torch.no_grad():
                for index, data in enumerate(dataLoader):
                    cFeature = featureMaker(data)
                    cFeature = cFeature.contiguous().view(-1, cFeature.size(2)//n_group)
                    Ck.append(cFeature)
                    if index > k:
                        break
            Ck = torch.cat(Ck, dim=0)
            N, D = Ck.size()
            indexes = torch.randperm(N)[:k]
            Ck = Ck[indexes].view(k, D)  #(1, k, D)
            # centers will be normalized from the very beginning, later only norm other stuff
            if norm_vec_len:
                # print("centers", Ck.shape)
                CkLengths = torch.sqrt((Ck*Ck).sum(1))
                # print("centers", CkLengths.shape)
                Ck = Ck / CkLengths.view(-1, 1)
                # CkLengths2 = torch.sqrt((Ck*Ck).sum(1))
                # print("********** cluster lengths after norm min, max", CkLengths2.min(), CkLengths2.max())
            Ck = Ck.view(1, k, D)
    else:
        Ck = start_clusters
        D = Ck.size(2)

    if perIterSize < 0:
        perIterSize = len(dataLoader)

    clusterStep = kMeanClusterStep(k, D, norm_vec_len=norm_vec_len).cuda()
    clusterStep = torch.nn.DataParallel(clusterStep)
    # CkLengths2 = torch.sqrt((Ck[0]*Ck[0]).sum(1))
    # print("********** cluster lengths after norm min, max", CkLengths2.min(), CkLengths2.max())
    clusterStep.module.Ck.copy_(Ck)
    # CkLengths2 = torch.sqrt((clusterStep.module.Ck[0]*clusterStep.module.Ck[0]).sum(1))
    # print("!!!!!!!!!!!!!!! set cluster lengths after norm min, max", CkLengths2.min().item(), CkLengths2.max().item())

    bar = progressbar.ProgressBar(maxval=MAX_ITER)
    bar.start()
    iter, stored = 0, 0
    if load and start_clusters is None and exists(join(save_dir, "checkpoint_last.pt")):
        iter = state_dict["iteration"]
        lastDiff = state_dict["lastDiff"]
        print(f"Continuing training from iteration {iter}. lastDiff: {lastDiff}")
    with torch.no_grad():
        while iter < MAX_ITER:
            start_time = time()
            Ck1 = torch.zeros(Ck.size()).cuda()
            nItemsClusters = torch.zeros(Ck.size(1),
                                         dtype=torch.long).cuda()
            for index, data in enumerate(dataLoader):
                cFeature = featureMaker(data).contiguous().view(-1, 1, D)
                locC, locN = clusterStep(cFeature)
                Ck1 += locC.sum(dim=0, keepdim=True)
                nItemsClusters += locN.sum(dim=0)
                ### If the training set is too big and we want to redude the number of item per iteration
                # stored += 1
                # if stored >= perIterSize:
                #     bar.update(iter)
                #     iter += 1
                #     stored = 0
                #     if iter >= MAX_ITER:
                #         break

            iter += 1
            bar.update(iter)

            nItemsClusters = nItemsClusters.float().view(1, -1, 1) + 1e-8
            Ck1 /= nItemsClusters

            if norm_vec_len:  # need to re-normalize, as mean of things of length 1 has length <= 1
                Ck1Lengths = torch.sqrt((Ck1*Ck1).sum(2))
                print("clustNorm", Ck1.shape, Ck1Lengths.shape, Ck1Lengths.view(*(Ck1Lengths.shape), 1).shape)
                Ck1 = Ck1 / Ck1Lengths.view(*(Ck1Lengths.shape), 1)

            lastDiff = (clusterStep.module.Ck - Ck1).norm(dim=2).max().item()
            nItems = int(nItemsClusters.sum().cpu().detach().item())
            # CkLengths2 = torch.sqrt((clusterStep.module.Ck[0]*clusterStep.module.Ck[0]).sum(1))
            # print("!!!!!!!!!!!!!!! prev cluster lengths after norm min, max", CkLengths2.min().item(), CkLengths2.max().item())
            # Ck1Lengths2 = torch.sqrt((Ck1[0]*Ck1[0]).sum(1))
            # print("!!!!!!!!!!!!!!!1 cluster lengths after norm min, max", Ck1Lengths2.min().item(), Ck1Lengths2.max().item())
            info=f"ITER {iter} done in {time()-start_time:.2f} seconds. nItems: {nItems}. Difference with last checkpoint: {lastDiff}"
            print(info)
            with open(join(save_dir, "training_logs.txt"), "a") as f:
                f.write(info+"\n")
            if save:
                info=f"Saving last checkpoint to {join(save_dir, 'checkpoint_last.pt')}"
                print(info)
                with open(join(save_dir, "training_logs.txt"), "a") as f:
                    f.write(info+"\n")
                out_state_dict = {}

                clusterModule = kMeanCluster(Ck1, norm_vec_len=norm_vec_len)
                out_state_dict["state_dict"] = clusterModule.state_dict()
                out_state_dict["n_clusters"] = Ck1.size(1)
                out_state_dict['dim'] = Ck1.size(2)
                out_state_dict["iteration"] = iter
                out_state_dict["lastDiff"] = lastDiff
                torch.save(out_state_dict, join(save_dir, "checkpoint_last.pt"))
                torch.save(out_state_dict, join(save_dir, f"checkpoint_{iter}.pt"))
                if exists(join(save_dir, f"checkpoint_{iter-save_last}.pt")):
                    remove(join(save_dir, f"checkpoint_{iter-save_last}.pt"))
            if lastDiff < EPSILON:
                print(
                    f"Clustering ended in {iter} iterations out of {MAX_ITER}")
                break
            clusterStep.module.Ck.copy_(Ck1)

    bar.finish()

    print(f"Clustering ended in {MAX_ITER} iterations out of {MAX_ITER}")
    print(f"Last diff {lastDiff}")
    if start_clusters is not None:
        nEmptyClusters = (nItemsClusters < 1).sum().item()
        print(f"{nEmptyClusters} empty clusters out of {k}")
    return clusterStep.module.Ck
