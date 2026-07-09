import ir_measures
from pyterrier_nuggetizer.measure._provider import NuggetEvalProvider
import pyterrier_nuggetizer
from pyterrier_nuggetizer.measure._util import RAGRunConverter

SUPPORTED_MEASURES = {'VitalScore', 'WeightedScore', 'AllScore'}


def measure_factory(attr: str, nuggetizer_provider: str):
    if attr in SUPPORTED_MEASURES:
        nuggetizer_provider.make_provider()
        return getattr(open_nuggetizer.measure._measures, attr)
