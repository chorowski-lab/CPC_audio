
import cpc.stats.stats_collector as sc
import cpc.stats.repr_diff_stat as repr_diff

# --valSetStats stat1:a,b,c_stat2
# --captureSetStats stat1:_stat2:p1,p2_stat3:p1

def getStatFromSpec(spec):
    specSplit = spec.split(":")
    statName, statArgs = specSplit[0], specSplit[1]
    statArgs = statArgs.split(",")
    assert statName in ("reprDiff,")
    if statName == "reprDiff":
        statArgs = repr_diff.ReprDiffStat.convertArgsFromStrings(*statArgs)
        return repr_diff.ReprDiffStat(*statArgs)

def constructStatCollectorFromSpecs(specs):
    specList = specs.split('_')
    collector = sc.StatsCollector()
    for spec in specList:
        collector.registerStat(getStatFromSpec(spec))
    return collector
