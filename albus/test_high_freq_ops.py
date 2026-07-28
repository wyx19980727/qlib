import sys
import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/albus/Python_Codes/qlib/examples")

from qlib import init
from qlib.constant import REG_HK
from qlib.data.data import ExpressionD, Cal
from qlib.data.ops import Operators
from highfreq.highfreq_ops import DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut

provider_uri_map = {
    "1min": "/home/albus/Python_Codes/qlib/qlib_data/silver_data_1min",
}
init(
    provider_uri=provider_uri_map,
    region=REG_HK,
    expression_cache=None,
    dataset_cache=None,
)
Operators.register([DayLast, FFillNan, BFillNan, Date, Select, IsNull, Cut])

inst = "00001"
freq = "1min"
start = "2026-05-22 09:30:00"
end = "2026-05-29 15:00:00"

def load(field):
    return ExpressionD.expression(inst, field, start, end, freq)

# --- DayLast ---
# import ipdb; ipdb.set_trace()
load('DayLast($close)')

# # --- FFillNan ---
# import ipdb; ipdb.set_trace()
load('FFillNan($close)')

# --- BFillNan ---
# import ipdb; ipdb.set_trace()
load('BFillNan($close)')

# --- Date ---
import ipdb; ipdb.set_trace()
load('Date($close)')

# --- IsNull ---
import ipdb; ipdb.set_trace()
load('IsNull($close)')

# --- Cut ---
import ipdb; ipdb.set_trace()
load('Cut($close, 10, -10)')

# --- Select ---
import ipdb; ipdb.set_trace()
load('Select(Gt($close, $open), $volume)')

# --- 组合 ---
import ipdb; ipdb.set_trace()
load('DayLast(FFillNan($close))')
load('DayLast(Cut($close, 5, -5))')
load('Select(IsNull($close), $close)')
