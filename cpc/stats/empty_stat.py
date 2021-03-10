
# this is a "template" for stat class, plus each stat must extend this
class Stat:

    def computeForBatch(self, batch):
        raise Exception("computeForBatch not implemented")

    def mergeStatResults(self, prev, current):
        raise Exception("mergeStatResults not implemented")

    def logStat(self, statValue, epochNr):
        raise Exception("logStat not implemented")
        # should return it's values in dict format, can also log somewhere
        # e.g. subclass can take 'where to log' as additional its-state arg

    def getStatName(self):
        raise Exception("getStatName not implemented")
        # has to differ if want to compute both at a time;
        # can be e.g. framewise_ctx_euclid_diff, framewise_ctx_cosine_diff dep. on Stat settings/state