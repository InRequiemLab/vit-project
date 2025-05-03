__version__ = "8.3.124"

import os
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  

from detection.models import  DETECT
from detection.utils import  SETTINGS
settings = SETTINGS
__all__ = (
    "DETECT",
    "settings",
)
