# -*- coding: utf-8 -*-
'''
used to store multiprocess function which should be outside the class
'''
import pandas as pd
import numpy as np
import logging
from scipy.stats import ks_2samp
import sys
sys.path.append(r"D:\Python_Code\ProT")


def multi_bin_info(series_var,series_label,size=0.1,sep_val=[],sep_size=0.1):
    # replace strange value to nan
    if series_var.dtype==str:
        series_var=series_var.replace({'':np.nan,'None':np.nan,'none':np.nan})

    # check dtype
    if not series_var.dtype in [int,float,complex]:
        try:
            series_var=series_var.astype(float)
        except ValueError:
            return 0

    # cut
    series_cut=separate(series_var,size=size,sep_val=sep_val,sep_size=sep_size)
    if not isinstance(series_cut,pd.Series):
        return 0

    detail,summary=create_info_table(series_cut,series_label)
    summary=summary+series_var.describe().tolist()
    return series_var.name,series_cut,detail,summary