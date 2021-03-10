
import cpc.stats.empty_stat as statTempl

class StatsCollector:

    def __init__(self):
        self.stats = []
        self.statValues = []
        self.zeroed = True
        self.statNames = set()

    def registerStat(self, stat):
        assert issubclass(type(stat), statTempl.Stat)
        assert stat.getStatName not in self.statNames
        self.statNames.add(stat.getStatName)
        self.stats.append(stat)

    def zeroStats(self):
        self.zeroed = True

    def batchUpdate(self, batch):
        if self.zeroed:
            self.statValues = [stat.computeForBatch(batch) for stat in self.stats]
        else:
            oldValues = self.statValues
            newValues = [stat.computeForBatch(batch) for stat in self.stats]
            self.statValues = [stat.mergeStatResults(prev, current) for stat, (prev, current) 
                                in zip (self.stats, zip(oldValues, newValues))]

    def dataLoaderUpdate(self, dataLoader):
        for batch in dataLoader:
            self.batchUpdate(batch)

    def logStats(self, epochNr):
        statLogs = {}
        for stat, statValue in zip(self.stats, self.statValues):
            statLogs.update({ stat.getStatName() + "_" + k: v for k, v in stat.logStat(statValue, epochNr).items()})
        return statLogs



