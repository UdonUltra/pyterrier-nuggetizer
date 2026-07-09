from ir_measures import measures

class _AllScore(measures.Measure):
    __name__ = 'AllScore'
    NAME = __name__
    PRETTY_NAME = 'All Score'
    SHORT_DESC = "Average of the scores for all nuggets in an answer"
    SUPPORTED_PARAMS = {
        'partial_rel': measures.ParamInfo(dtype=int, default=1, desc='maximum partial value (inclusive)'),
        'strict': measures.ParamInfo(dtype=bool, default=False, desc='Exclude nuggets partially supported in measure'),
    }

AllScore = _AllScore()
measures.register(AllScore)


class _VitalScore(measures.Measure):
    __name__ = 'VitalScore'
    NAME = __name__
    PRETTY_NAME = 'Vital Score'
    SHORT_DESC = "Average of the scores for all nuggets in an answer"
    SUPPORTED_PARAMS = {
        'rel': measures.ParamInfo(dtype=int, default=1, desc='Minimum vital value (inclusive)'),
        'partial_rel': measures.ParamInfo(dtype=int, default=1, desc='maximum partial value (inclusive)'),
        'strict': measures.ParamInfo(dtype=bool, default=True, desc='Exclude nuggets partially supported in measure'),
    }

VitalScore = _VitalScore()
measures.register(VitalScore)


class _WeightedScore(measures.Measure):
    __name__ = 'WeightedScore'
    NAME = __name__
    PRETTY_NAME = 'Weighted Score'
    SHORT_DESC = "Weighted average of the scores for all nuggets in an answer"
    SUPPORTED_PARAMS = {
        'rel': measures.ParamInfo(dtype=int, default=1, desc='Minimum vital value (inclusive)'),
        'partial_rel': measures.ParamInfo(dtype=int, default=1, desc='maximum partial value (inclusive)'),
        'partial_weight': measures.ParamInfo(dtype=float, default=0.5, desc='Weight for partial support nuggets'),
    }


WeightedScore = _WeightedScore()
measures.register(WeightedScore)


# debug printing measures registry
#def print_measures_registry():
#    print("Registered measures:")
#    for measure in measures.registry:
#        print(measure)
#    print("Registered measures with details:")
#print_measures_registry()
