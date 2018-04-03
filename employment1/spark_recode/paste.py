import pandas as pd
import numpy as np
import multiprocessing
from scipy.stats import ks_2samp
import multiprocessing
import logging
import time
from multiprocessing import Pool
import functools

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s%(levelname)s %(message)s',
                datefmt='%d %b %Y %H:%M:%S',
                filename='test.log',
                filemode='wb')
logger = logging.getLogger("DataProcess")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',"%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.handlers=[]
logger.addHandler(ch)

def separate(series, size=0.1, sep_val=[], sep_size=0.1):
    """
    Missing value should be set as -np.inf outside this function
    inf is always in separate value
    """
    logger.info("binning with "+series.name)
    # check level
    VC=series.value_counts()
    if len(VC) < 1.0 / size + 2:
        return series

    
    series=series.fillna(-np.inf)
    # check nan. Nan must be sep.
    if series.isin([-np.inf]).any() and -np.inf not in sep_val:
        sep_val.append(-np.inf)
    # check sep option
    if sep_val and sep_size:
        threshold=len(series)*size
        oversize_val=[VC.index.values.tolist()[ind] for ind, val in enumerate(VC.values.tolist()) if val > threshold]
        sep=list(set(sep_val+oversize_val))
    elif sep_val:
        sep=sep_val
    elif sep_size:
        threshold=len(series)*size
        sep=[VC.index.values.tolist()[ind] for ind, val in enumerate(VC.values.tolist()) if val > threshold]
    else:
        return pd.qcut(series,int(1.0/size))
        
    # extract
    mask=series.isin(sep)
    cut_part = series[~mask]
    sep_part = series[mask]
    cut_values_total = sorted(cut_part)
    remain_size = len(cut_part) * 1.0 / len(series)
    cut_index = list(np.linspace(0, len(cut_part) - 1, int(remain_size/size) + 1))
    cut_values = [-np.inf] + [cut_values_total[int(a)] for a in cut_index][1:-1] + [np.inf]
    after_cut = pd.cut(cut_part, cut_values)
    res = pd.concat([after_cut, sep_part], axis=0).astype("category")
    combine_categories = sep + list(after_cut.cat.categories.values)
    res = res.cat.set_categories(combine_categories).sort_index()
    res.name=series.name
    return res

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

class DataProcess(object): 
    """
    a=DataProcessing(df)
    a.method()

    Thus, self is a pd.DataFrame()
    """
    def __init__(self,df,key,label):
        self.df=df
        self.var_names=[i for i in df.columns.values.tolist() if i not in key+[label]]
        self.key=key #list of str
        self.label=label  #str
        self.n_sample=df.shape[0]
        self.summary={}
        self.summary_dict={}
        self.detail={}
        

    def bin_info_var(self,var,size=0.1,sep_val=[],sep_size=0.1):
        series=self.df[var]
        # replace strange value to nan
        if series.dtype==str:
            series=series.replace({'':np.nan,'None':np.nan,'none':np.nan})
        # check dtype
        if not series.dtype in [int,float,complex]:
            try:
                series=series.astype(float)
            except ValueError:
                return 1
        # cut and info
        var_cut=separate(series,size=size,sep_val=sep_val,sep_size=sep_size)
        detail,summary=create_info_table(var_cut,self.df[self.label])

        # add describe() to summary list
        summary=summary+series.describe().tolist()
        end=time.time()
        return detail,summary

    def bin_info_df(self,size=0.1,sep_val=[],sep_size=0.1):
        start = time.time()
        self.detail={}
        self.summary_dict={}
        for var in self.var_names:
            res=self.bin_info_var(var,size=size,sep_val=sep_val,sep_size=sep_size)
            #can not change dtype to float
            if res==1:
                break
            else:
                self.detail[var]=res[0]
                self.summary_dict[var]=res[1]
        index=['IV','KS','levels','iv_monotonity','max_prop','count','mean','std','min','25%','50%','75%','max']
        self.summary=pd.DataFrame.from_dict(self.summary_dict)
        self.summary.index=index
        self.summary=self.summary.transpose()

    def bin_info(self,size=0.1,sep_val=[],sep_size=0.1,cores=8):
        start = time.time() 
        names=self.var_names
        pool = multiprocessing.Pool(cores)
        result=pool.map_async(functools.partial(multi_bin_info,series_label=self.df[self.label]), [df[i] for i in names])
        pool.close()
        pool.join()
        logger.info("Parallel mapping. Done")
        res=result.get()
        logger.info("Result getting. Done")

        logger.info("Extract Resulting")
        series_cut_list=[]
        for i in res:
            if i:
                self.detail[i[0]]=i[2]
                self.summary_dict[i[0]]=i[3]
                series_cut_list.append(i[1])

        logger.info("Generating cut")
        self.cut = pd.concat(series_cut_list,axis=1,join='inner')
        self.cut = pd.concat([self.cut,self.df[self.label]],axis=1)
        logger.info("CUT DONE")

        logger.info("Generating summary")
        self.summary = pd.DataFrame.from_dict(self.summary_dict)
        self.summary.index=['IV','KS','levels','iv_monotonity','max_prop','count','mean','std','min','25%','50%','75%','max']
        self.summary=self.summary.transpose()


        end = time.time()
        print 'Time consumed：',end - start

    def debug_bin_info(self):
        d={}
        for i in self.var_names:
            d[i]=multi_bin_info(self.df[i],self.df[self.label])
        return d





df=pd.read_csv("~/wacai/wacainew_170308.csv")
a=DataProcess(df,['appno','debit_card_no','device_id','id','mobile','time','i_province'],'code')
#a.debug_bin_info()
a.bin_info()
