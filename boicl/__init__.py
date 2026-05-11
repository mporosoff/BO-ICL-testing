from .version import __version__
from .asktell import AskTellFewShotTopk
from .asktellFinetuning import AskTellFinetuning
from .asktellNearestNeighbor import AskTellNearestNeighbor
from .datasets import build_ocm_dataset, write_ocm_dataset
from .pool import Pool
from .tool import BOICLTool


__all__ = [
    "AskTellFewShotTopk",
    "AskTellFinetuning",
    "AskTellNearestNeighbor",
    "build_ocm_dataset",
    "write_ocm_dataset",
    "Pool",
    "BOICLTool",
]

try:
    from .asktellGPR import AskTellGPR
    from .asktellRidgeRegression import AskTellRidgeKernelRegression
except ImportError:
    pass
else:
    __all__.extend(["AskTellGPR", "AskTellRidgeKernelRegression"])
