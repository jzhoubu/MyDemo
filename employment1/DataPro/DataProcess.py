# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import logging
from scipy.stats import ks_2samp
import time
import sys
sys.path.append(r"D:\Python_Code\ProT")
from DataPro import logger
from DataPro.bin import *
from DataPro.info import *
from DataPro.multi_cores import *


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
        return var_cut,detail,summary

    def bin_info_win(self,size=0.1,sep_val=[],sep_size=0.1):
        start = time.time()
        cut_list=[]
        self.detail={}
        self.summary_dict={}
        for var in self.var_names:
            res=self.bin_info_var(var,size=size,sep_val=sep_val,sep_size=sep_size)
            #can not change dtype to float
            if res==1:
                break
            else:
                cut_list.append(res[0])
                self.detail[var],self.summary_dict[var]=res[1:3]
        self.summary=pd.DataFrame.from_dict(self.summary_dict)
        self.summary.index=['IV','KS','levels','iv_monotonity','max_prop','count','mean','std','min','25%','50%','75%','max']
        self.summary=self.summary.transpose().sort_values(by='IV',ascending=False)
        self.cut = pd.concat(cut_list,axis=1,join='inner')
        end=time.time()
        print 'Time consumed：',end - start



    def bin_info(self,size=0.1,sep_val=[],sep_size=0.1,cores=8):
        start = time.time() 
        names=self.var_names
        pool = multiprocessing.Pool(cores)
        result=pool.map_async(functools.partial(multi_bin_info,series_label=self.df[self.label]), [df[i] for i in names])
        pool.close()
        pool.join()
        logger.info("Parallel arrange. Done")
        res=result.get()
        logger.info("Get result. Done")

        logger.info("Extract result. Done")
        series_cut_list=[]
        for i in res:
            if i:
                self.detail[i[0]]=i[2]
                self.summary_dict[i[0]]=i[3]
                series_cut_list.append(i[1])
        logger.info("Generating dict. Done")
        
        self.cut = pd.concat(series_cut_list,axis=1,join='inner')
        self.cut = pd.concat([self.cut,self.df[self.label]],axis=1)
        logger.info("Generating cut. Done")

        self.summary = pd.DataFrame.from_dict(self.summary_dict)
        self.summary.index=['IV','KS','levels','iv_monotonity','max_prop','count','mean','std','min','25%','50%','75%','max']
        self.summary=self.summary.transpose().sort_values(by='IV',ascending=False)
        logger.info("Generating summary. Done")


        end = time.time()
        print 'Time consumed：',end - start


