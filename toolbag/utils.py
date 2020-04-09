import numpy as np

def sliding_window(ini, end, width, step):
    '''Returns a vector with a sliding window.'''

    return [(t, (t+width)) for t in np.arange(ini, end - width + 1, step)]


import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Note that I'm ignoring clipping and other edge cases here.
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)

from datetime import date
def today(invert=False):
    if invert:
        return '_'.join(str(date.today()).split('-'))
    else:
        return '_'.join(str(date.today()).split('-')[::-1])

def find_pairs(df, groups, condition_col, conditions=None, only_first=False, flatten=False):
    ixs = []
    for group, dff in df.groupby(groups):
        ixs_cond = []
        if conditions is None:
            conditions = df[condition_col].unique()

        for cond in conditions:
            ixs_cond.append(dff[dff[condition_col]==cond].index.to_list())
        
        all_nonempty = np.all([len(x) > 0 for x in ixs_cond])
        if all_nonempty:
            if only_first:
                ixs.append([x[0] for x in ixs_cond])
            else:
                ixs.append(ixs_cond)
    if flatten:
        if only_first:
            ixs = [x for y in ixs for x in y]
        else:
            ixs = [x + y for x,y in ixs]
            ixs = [x for y in ixs for x in y]
        ixs = np.sort(ixs)
        
    return ixs