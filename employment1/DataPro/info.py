# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import sys
sys.path.append(r"D:\Python_Code\ProT")
from DataPro import logger


def create_info_table(series,label_s):
    df=pd.concat([series,label_s],axis=1)
    var=series.name
    label=label_s.name
    group = df.groupby(by=[var])
    n_group=group[var].count()
    n_bad = df[label].sum()
    n_good = df[label].count() - n_bad
    n_all = df.shape[0]
    
    # prop
    prop = n_group.div(n_all)
    # Bad_count
    Bad_count = group[label].sum()
    # Good_count
    Good_count = n_group - Bad_count
    # Bad%
    Bad_r = Bad_count.div(n_bad)
    # Good%
    Good_r = Good_count.div(n_good)
    # Bad%_inside
    Bad_r_inside = Bad_r.div(Good_r + Bad_r)
    # Good%_inside
    Good_r_inside = 1 - Bad_r_inside
    # WOE
    woe = np.log(Good_r) - np.log(Bad_r)
    # IV
    weight = Good_r - Bad_r
    iv = weight * woe
    
    
    info={'prop':prop,\
          'Bad_count':Bad_count,\
          'Good_count':Good_count,\
          'Bad_r':Bad_r,\
          'Good_r':Good_r,\
          'Bad_r_inside':Bad_r_inside,\
          'Good_r_inside':Good_r_inside,\
          'woe' : woe,\
          'weight' : weight,\
          'iv':iv
          }
    info=pd.DataFrame.from_dict(info)
    info['iv_cum']= info['iv'].cumsum()
    info=info.loc[info.prop>0,:]
    levels=range(info.shape[0])
    info['KS'] = ks_2samp(list(np.repeat(levels,info['Bad_count'].map(int))), list(np.repeat(levels,info['Good_count'].map(int)))).statistic
    info=info[['iv_cum','KS','prop','Bad_count','Good_count','Bad_r','Good_r','Bad_r_inside','Good_r_inside','woe','weight','iv']]

    ################SUMMARY###############
    IV=info.iv_cum.tolist()[-1]
    # KS
    KS=info.KS.tolist()[-1]
    # bins
    n_bins=info.shape[0]
    # iv_monotonity
    iv_diff=np.diff(info.iv.tolist())
    iv_monotonity=all(iv_diff>0) or all(iv_diff<0)
    # max_prop
    max_prop=np.max(info.prop.tolist())
    summary=[IV,KS,n_bins,iv_monotonity,max_prop]
    return info,summary


