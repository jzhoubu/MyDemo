import pandas as pd
import numpy as np
import sys
sys.path.append(r"D:\Python_Code\ProT")
from DataPro import logger




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


def greedyks(data,series, label, size):
    data = data.loc[:, [series, label]]
    n = data.shape[0]
    sort = sorted(data[series])
    val2q = {val: ind * 1.0 / n for ind, val in enumerate(sort)}
    q2val = {k: v for v, k in val2q.items()}
    qlist = sorted(q2val.keys())

    class collect:
        q = [0, 1]

    def _kspoint(data, series, label):
        data0 = np.sort(data.loc[data[label] == 0, series])
        data1 = np.sort(data.loc[data[label] == 1, series])
        if data0.shape[0]==0 or data1.shape[0]==0:
            return 0
        data_all = np.concatenate([data0, data1])
        cdf0 = np.searchsorted(data0, data_all, side='right') / (1.0 * data0.shape[0])
        cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * data1.shape[0])
        return data_all[np.argmax(np.absolute(cdf0 - cdf1))]

    def _check(q):
        return np.min([np.abs(i - q) for i in collect.q]) > size

    def _next(q, d):
        if d == 'right':
            temp = [i for i in qlist if i > q + size]
            ans = np.min(temp) if len(temp) > 1 else 1
        else:
            temp = [i for i in qlist if i < q - size]
            ans = np.max(temp) if len(temp) > 1 else 0
        return ans

    def _kscut(data, father, direction='both'):
        # print "start _kscut"
        if direction == 'both':
            # print "###################both########################"
            val = _kspoint(data, series, label)
            # print "both ksp",val
            q = val2q.get(val)
            # print "both q",q
            collect.q.append(q)
            # print "collector",collect.q
            # print "###################right########################"
            _kscut(data.loc[data[series] > val, :], q, 'right')
            # print "###################left########################"
            _kscut(data.loc[data[series] < val, :], q, 'left')

        elif direction == 'right':
            # print "right ks val,q=",val,q
            next = _next(father, 'right')
            if not _check(next):
                # print "next to the end. STOP",next
                return 0
            val = _kspoint(data, series, label) 
            q = val2q.get(val)
            if _check(q):
                # print "ks pass"
                collect.q.append(q)
                _kscut(data.loc[data[series] > val, :], q, 'right')
            else:
                # print "ks not pass"
                collect.q.append(next)
                next_val = q2val.get(next)
                # print "append next ",next,next_val
                _kscut(data.loc[data[series] > next_val, :], q, 'right')

        elif direction == 'left':
            next = _next(father, 'left')
            if not _check(next):
                return 0
            val = _kspoint(data, series, label)  # data here is local seriesiable
            q = val2q.get(val)
            if _check(q):
                collect.q.append(q)
                _kscut(data.loc[data[series] < val, :], q, 'left')
            else:
                collect.q.append(next)
                next_val = q2val.get(next)
                _kscut(data.loc[data[series] < next_val, :], q, 'left')

    _kscut(data, 'both')
    collect.q.remove(0)
    collect.q.remove(1)
    cp_val = map(q2val.get, collect.q)
    cp_val = sorted([-np.inf] + cp_val + [np.inf])
    res = pd.cut(data[series], cp_val)
    return res

def sepsparse(series, size, sep_val=[], sep_size=False, sparse=1):
    """
    Separate oversize values and then cut the remain values with a sparse dict.
    size:      bin size
    sep:       list type. Values that need to calculate separately.
    sep_size:  Values proportion over sep_size with be append in sep list
    sparse:    Granular size used in calculate
    """
    def check_dtype(Series):
        if Series.dtype==str:
            return pd.to_numeric(Series,errors='raise')
        return Series


    def find_oversize(Series,threshold,VC):
        oversize_val=[VC.index.values.tolist()[ind] for ind, val in enumerate(VC.values.tolist()) if val > threshold]
        return oversize_val


    def sparse_bin(Series, step, sparse_size=1):
        total_n = len(Series)
        sparse_index = range(total_n) if sparse_size == 1 else map(int, list(np.linspace(0, total_n - 1, sparse_size)))
        sparse_n = len(sparse_index)
        step = step * 1.0 / sparse_size
        values = sorted(Series.values.tolist())
        sparse_values = [values[i] for i in sparse_index]
        val2ind = {v: i for i, v in enumerate(sparse_values)}
        ind2val = {v: k for k, v in val2ind.iteritems()}
        LastIndex = ind2val.keys()
        cut_index = [0]
        while cut_index[-1] <= sparse_n - step:
            cut_index.append([i for i in LastIndex if i >= (cut_index[-1] + step)][0])
        # print cut_index
        cut_value = list(map(ind2val.get, cut_index))
        cut_value = [-np.inf] + cut_value[1:-1] + [np.inf]
        return pd.cut(Series, cut_value)
    
    
    series = check_dtype(series)
    VC = series.value_counts()
    step = len(series) * size
    if sep_size and sep_val:
        sep = list(set(sep_val + find_oversize(series, len(series) * size, VC)))
    elif sep_size:
        sep = find_oversize(series, len(series) * size, VC)
    elif sep_val:
        sep = sep_val
    else:
        return sparse_bin(series, step=step, sparse_size=sparse)

    mask = series.isin(sep)
    sep_part = series[mask]
    cut_part = series[~mask]
    after_cut = sparse_bin(Series=cut_part, step=step, sparse_size=sparse)
    res = pd.concat([sep_part, after_cut], axis=0).astype("category")

    combine_categories = sep + list(after_cut.cat.categories.values)
    res = res.cat.set_categories(combine_categories)
    return res