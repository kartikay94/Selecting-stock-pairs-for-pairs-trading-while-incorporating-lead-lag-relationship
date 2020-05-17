



import numpy as np


# rtscd = rtscd.values

import copy

# # rtscd_copy = copy.copy(rtscd[:])
import pandas as pd

from scipy.cluster import hierarchy as hr

import scipy
# import matplotlib.pyplot as plt


import matplotlib as mt
mt.use('Agg')

# from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import fcluster

import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
import fastdtw

# pip install pydtw
# rtscd.shape
# # dm.method3 = jj

import scipy
import numpy  as np
import pandas as pd





def transform_ts(x=None , p=100 ):
    '''
    x = rtscd[:,0]
    '''
    series_ln = x.shape[0]
    x_new = []

    for i in range( series_ln + 1 - p ):
        x_new.append( x[i:(i+p)].tolist() )   # x[i:(i+p)].shape[0]

    x_new = np.array( x_new )

    return x_new




def inner_cdtw_distance( x_new_i, y_new_j ):
    '''
    x_new_i = x_new[0]
    y_new_j = y_new[0]
    '''
    # remember it is not absolute, invert plus minus
    # remember it is not absolute, invert plus minus
    dist_tmp = np.sum( x_new_i*y_new_j )/( (np.sum( x_new_i**2 )**0.5)*(np.sum( y_new_j**2 )**0.5) )
    # assert dist_tmp is not None
    # if dist_tmp is None:
    #     dist_tmp = 0

    # # # final_dist = 2/( 1 + np.e**(4*dist_tmp) )
    final_dist = 2*(1 - dist_tmp ) # this is preferred as a metric closer
    # # # final_dist = (2*(1 - dist_tmp ))**0.5

    return final_dist





def wping_paths_trnsfrmd(os1, os2, window=None, max_dist=None,
                  max_step=None, max_length_diff=None, penalty=None, psi=50,  p = 50, m = 100, n = 18,  Threshold = 0.35 , jobs = 1, rtrn = 0):
    """
    Dynamic Time Warping.

    The full matrix of all warping paths is build.

    :param s1: First sequence
    :param s2: Second sequence
    :param window: see :meth:`distance`
    :param max_dist: see :meth:`distance`
    :param max_step: see :meth:`distance`
    :param max_length_diff: see :meth:`distance`
    :param penalty: see :meth:`distance`
    :param psi: see :meth:`distance`
    :returns: (DTW distance, DTW matrix)

    window: Only allow for shifts up to this amount away from the two diagonals.
    max_dist: Stop if the returned distance measure will be larger than this value.
    max_step: Do not allow steps larger than this value.
    max_length_diff: Return infinity if difference in length of two series is larger.
    penalty: Penalty to add if compression or expansion is applied (on top of the distance).
    psi: Psi relaxation to ignore begin and/or end of sequences (for cylical sequencies) [2].

    """

    s1 = transform_ts(os1, p )
    s2 = transform_ts(os2, p )

    r, c = len(s1), len(s2)
    if max_length_diff is not None and abs(r - c) > max_length_diff:
        return np.inf
    if window is None:
        window = max(r, c)
    if not max_step:
        max_step = np.inf
    else:
        max_step *= max_step
    if not max_dist:
        max_dist = np.inf
    else:
        max_dist *= max_dist
    if not penalty:
        penalty = 0
    else:
        penalty *= penalty
    if psi is None:
        psi = 0
    dtw = np.full((r + 1, c + 1), np.inf)
    # dtw[0, 0] = 0
    for i in range(psi + 1):
        dtw[0, i] = 0
        dtw[i, 0] = 0
    last_under_max_dist = 0
    i0 = 1
    i1 = 0
    for i in range(r):
        if last_under_max_dist == -1:
            prev_last_under_max_dist = np.inf
        else:
            prev_last_under_max_dist = last_under_max_dist
        last_under_max_dist = -1
        i0 = i
        i1 = i + 1
        # print('i =', i, 'skip =',skip, 'skipp =', skipp)
        # jmin = max(0, i - max(0, r - c) - window + 1)
        # jmax = min(c, i + max(0, c - r) + window)
        # print(i,jmin,jmax)
        # x = dtw[i, jmin-skipp:jmax-skipp]
        # y = dtw[i, jmin+1-skipp:jmax+1-skipp]
        # print(x,y,dtw[i+1, jmin+1-skip:jmax+1-skip])
        # dtw[i+1, jmin+1-skip:jmax+1-skip] = np.minimum(x,
        #                                                y)
        for j in range(max(0, i - max(0, r - c) - window + 1), min(c, i + max(0, c - r) + window)):
            # print('j =', j, 'max=',min(c, c - r + i + window))
            # # # # d = (s1[i] - s2[j])**2
            d = inner_cdtw_distance( s1[i], s2[j] )
            # print(d)
            if max_step is not None and d > max_step:
                continue
            # print(i, j + 1 - skip, j - skipp, j + 1 - skipp, j - skip)
            dtw[i1, j + 1] = d + min(dtw[i0, j],
                                     dtw[i0, j + 1] + penalty,
                                     dtw[i1, j] + penalty)
            # dtw[i + 1, j + 1 - skip] = d + min(dtw[i + 1, j + 1 - skip], dtw[i + 1, j - skip])
            if max_dist is not None:
                if dtw[i1, j + 1] <= max_dist:
                    last_under_max_dist = j
                else:
                    dtw[i1, j + 1] = np.inf
                    if prev_last_under_max_dist < j + 1:
                        break
        if max_dist is not None and last_under_max_dist == -1:
            # print('early stop')
            # print(dtw)
            return np.inf, dtw
    # print(dtw)
    dtw = np.sqrt(dtw)
    # print(dtw)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    mir_i = 0
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if psi == 0:
        d = dtw[i1, min(c, c + window - 1)]
    else:
        ir = i1
        ic = min(c, c + window - 1)
        vr = dtw[ir-psi:ir+1, ic]
        vc = dtw[ir, ic-psi:ic+1]
        mir = np.argmin(vr)
        mic = np.argmin(vc)
        if vr[mir] < vc[mic]:
            dtw[ir-psi+mir+1:ir+1, ic] = -1
            d = vr[mir]
            mir_i = 1
        else:
            dtw[ir, ic - psi + mic + 1:ic+1] = -1
            d = vc[mic]

    no_terms = np.shape(s1)

    if rtrn == 0:
        return d
    elif rtrn == 1:
        '''
        vr is the row in probably last column
        vc is probably the column in last row
        mir is the required minimum value there
        mic is the requires minimum value of column in the last row

        where this number comes, in that linear segment rest values are made 0
        '''
        return [d, dtw, vr , vc , mir, mic, mir_i, no_terms[0], no_terms[1] ]











def wping_path(from_s, to_s, **kwargs):
    """Compute warping path between two sequences."""
    dist, paths = warping_paths(from_s, to_s, **kwargs)
    path = best_path(paths)
    return path


def wping_amount(path):
    """
        Returns the number of compressions and expansions performed to obtain the best path.
        Can be used as a metric for the amount of warping.
        :param path: path to be tested
        :returns number of compressions or expansions
    """
    n = 0
    for i in range(1, len(path)):
        if path[i - 1][0] + 1 != path[i][0] or path[i - 1][1] + 1 != path[i][1]:
            n += 1

    return n



def warp(from_s, to_s, **kwargs):
    """Warp a function to optimally match a second function.
    Same options as :meth:`warping_paths`.
    """
    path = wping_path(from_s, to_s, **kwargs)
    from_s2 = np.zeros(len(to_s))
    from_s2_cnt = np.zeros(len(to_s))
    for r_c, c_c in path:
        from_s2[c_c] += from_s[r_c]
        from_s2_cnt[c_c] += 1
    from_s2 /= from_s2_cnt
    return from_s2, path


def best_path(paths):
    """Compute the optimal path from the nxm warping paths matrix."""
    i, j = int(paths.shape[0] - 1), int(paths.shape[1] - 1)
    p = []
    if paths[i, j] != -1:
        p.append((i - 1, j - 1))
    while i > 0 and j > 0:
        c = np.argmin([paths[i - 1, j - 1], paths[i - 1, j], paths[i, j - 1]])
        if c == 0:
            i, j = i - 1, j - 1
        elif c == 1:
            i = i - 1
        elif c == 2:
            j = j - 1
        # print('i=',i, 'j=',j)
        if paths[i, j] != -1:
            p.append((i - 1, j - 1))
    p.pop()
    p.reverse()
    return p


def best_path2(paths):
    """Compute the optimal path from the nxm warping paths matrix."""
    m = paths
    path = []
    r, c = m.shape
    r -= 1
    c -= 1
    v = m[r, c]
    path.append((r - 1, c - 1))
    while r > 1 or c > 1:
        r_c, c_c = r, c
        if r >= 1 and c >= 1 and m[r - 1, c - 1] <= v:
            r_c, c_c, v = r - 1, c - 1, m[r - 1, c - 1]
        if r >= 1 and m[r - 1, c] <= v:
            r_c, c_c, v = r - 1, c, m[r - 1, c]
        if c >= 1 and m[r, c - 1] <= v:
            r_c, c_c, v = r, c - 1, m[r, c - 1]
        path.append((r_c - 1, c_c - 1))
        r, c = r_c, c_c
    path.reverse()
    return path










def transformed(x=None , p=100 ):
    '''
    ts['x(i)'] ts['target']
    ts[col] for col in pred_cols
    t2_pred


    x = np.random.random_sample((400,3))
    '''

    series_ln = x.shape[0]
    x_new = []

    for i in range( series_ln + 1 - p ):
        x_new.append( x[i:(i+p)].tolist() )   # x[i:(i+p)].shape[0]e[0]

    x_new = np.array( x_new )

    return x_new

































































import pandas as pd
import numpy as np

# # WARPing_path_001 is the correct old file

import numpy as np
# import matplotlib.pyplot as plt
import time
from thermal_optimal_path.lattice import partition_function
from thermal_optimal_path.statistics import average_path

import pandas as pd
import numpy as np
import numpy as np
# import matplotlib.pyplot as plt
import time

from scipy.stats import t
from thermal_optimal_path.lattice import partition_function
from thermal_optimal_path.statistics import average_path

from fastdtw import fastdtw
import fastdtw

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np




def standardise(ts):
    ts -= np.mean(ts)
    return ts / np.std(ts)

np.random.seed(110)

# # # # # # # # # # # # # # # # # # # # # # # #
# At higher error_x values this function gives nan at higher values
# This is because we divide by the exponential of the error function
# # # # # # # # # # # # # # # # # # # # # # # #


def get_actual_path( W , ll ):
    actual_path = []

    for inx,llv in enumerate(ll):
        # print(inx,llv)
        for i in range(W*inx, W*(inx+1)):
            actual_path.append(llv)

    return actual_path


def get_lead_DTW_euclid(x, y):
    # x = xy[:,0]
    # y = xy[:,1]
    d, paths = dtw.warping_paths(x, y, window=None, psi=None)
    best_path = dtw.best_path(paths)
    # # dtwvis.plot_warpingpaths( x,  y, paths, best_path)
    lead_E = create_lead_from_path(best_path, x, 0 )
    return (lead_E,d)






def create_time_series(L=300,W=50,ll=[0,5,10,-10,-5,0],erx = 1, a=0.8,b=0.7, f = 0.2 ):
    '''
    W*len(ll) = L
    '''
    error_x = erx
    error_y = error_x*f
    x = np.zeros(L)
    y = np.zeros(L)

    for i in range(1, L):
        x[i] = b * x[i-1] + error_x * np.random.randn(1)

    for inx,llv in enumerate(ll):
        # print(inx,llv)
        for i in range(W*inx, W*(inx+1)):
            y[i] = a * x[i-llv] + error_y * np.random.randn(1)

    x = standardise(x)
    y = standardise(y)

    return np.transpose( np.vstack((x,y)) )



def augmented_zero_ts(x_original, y_original, no_of_zeroes = 8):

    xy0 = np.stack([x_original, y_original], axis=1)
    xy1 = np.append([[0,0]]*no_of_zeroes, xy0 , axis=0)

    xy2 = np.append(xy1, [[0,0]]*no_of_zeroes, axis=0)
    xy2 = np.array(xy2, dtype = 'double')
    # first col is spot, second is option

    x = xy2[:,0]
    y = xy2[:,1]
    return x,y



# X ///; np.r_[path][:,0]
def create_lead_from_path(path , x , no_of_zeroes ):
        '''
        How to get a lead-lag values of smae size as of x and y
        if the start or end is shifted add continuous values values for it
        else
        Presently, we add the first laed-lag values of repeated x path
        ideally it should start with 0
        means x= no_of
        '''
        # start_shift = 10
        # end_shift = 10
        start_shift = path[0][0]
        end_shift = len(x) - 2*no_of_zeroes - ( path[-1][0] +1 )
        if start_shift >0:
            path = np.append([(np.nan, np.nan)]*start_shift , path, axis=0)

        if end_shift >0:
            path = np.append(path,[(np.nan, np.nan)]*end_shift  , axis=0)

        path = path.copy()

        dups1 = np.r_[ 1, np.r_[path][1:,0] - np.r_[path][:-1,0] ]

        # dups = [i for i in dups if np.isnan()]
        dups = np.array([])
        for i in dups1:
            if np.isnan(i):
                dups = np.append(dups, 1)
            else:
                dups = np.append(dups, i)

        taoi = np.r_[path][:,1] - np.r_[path][:,0]

        np.r_[path]
        taolst = np.concatenate( (   np.transpose( np.r_[path] ), np.array([taoi]) ,  np.array([dups] )) , axis =0 )

        taolst = np.transpose(taolst)
        lead = []
        for i in taolst:
            if i[3] == 1:
                lead.append(i[2])

        # # # print(len(lead))

        # # # # # # # # # # # # # # # # # # # # # # #
        # # this is unnecessay now
        # nans = int( p_val/2 )
        # lead100 = np.r_[[np.nan]*nans, lead, [np.nan]*nans ]
        # # # # # # # # # # # # # # # # # # # # # # # # #
        return lead






def get_top(x, y, temperature):
    g2 = partition_function(x, y, temperature)
    average_path(g2)
    avg2 = average_path(g2)[::2]
    return list(avg2)



def get_lead_from_zero_p(xy, no_of_zeroes = 8):
    '''
    lead based on p value
    '''
    p_val = 2*no_of_zeroes + 1

    x,y = augmented_zero_ts(xy, no_of_zeroes = no_of_zeroes)

    print(time.time())
    t1 = time.time()


    dst = wping_paths_trnsfrmd(x.copy(), y.copy(), window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi= p_val,  p = p_val, m = 25, n = 10,  Threshold = 0.35 , jobs = 1, rtrn = 1)

    # dst[0]
    # dst[1]

    t2 = time.time() - t1
    print(t2/60)

    path_ = best_path(dst[1])


    lead = create_lead_from_path(path = path_ , x = x , no_of_zeroes = no_of_zeroes )
    return lead


from scipy.stats import t



def best_z_val(x_original, y_original , leadpdf, zeros_list ):
    xy = np.stack([x_original, y_original], axis=1)
    x_o = xy[:,0]
    y_o = xy[:,1]
    leadpdf['x_o'] = x_o
    leadpdf['y_o'] = y_o
    leadpdf['sno'] = range(len(x_o))
    for z in zeros_list:
        leadpdf['l_'+ str(z)] =  leadpdf['sno'] - leadpdf[z]

    for z in zeros_list:
        # quick fix
        # squashing index values btw limits
        tmp_lag = [ x_o[min(max(0,int(j)), (len(x_o)-1) )] for j in leadpdf['l_'+ str(z)].values ]
        leadpdf['x_l_'+ str(z)] = tmp_lag


    cor_z = []
    for z in zeros_list:
        cor_v = np.corrcoef(leadpdf['x_l_'+ str(z)].values , leadpdf['y_o'].values )[0,1]
        # print(z, cor_v)
        euclid_v = (np.sum( (leadpdf['x_l_'+ str(z)].values - leadpdf['y_o'].values)**2 ))**0.5
        mad_v = np.sum( abs(leadpdf['x_l_'+ str(z)].values - leadpdf['y_o'].values) )

        cor_z.append([z,cor_v,euclid_v,mad_v ])

    cor_z = np.array(cor_z)
    # # # print(cor_z)

    # best_z = int(cor_z[np.argmax(cor_z[:,1] ),0])
    # best_z = int(cor_z[np.argmax(cor_z[:,1] ),0])
    best_z = int(cor_z[np.argmin(cor_z[:,2] ),0])

    leadpdf.columns
    cor_z = pd.DataFrame(cor_z)
    cor_z.columns = ['z','cor_v','euclid_v','mad_v']
    cor_z['index'] = np.array( cor_z.z.values, dtype='int' )
    cor_z.set_index('index', inplace=True)
    
    return (best_z, cor_z, leadpdf)






def get_lead_from_zero(x_original, y_original, zeros_list = [12,25,50],  window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=50,  p = 50, m = 100, n = 18,  Threshold = 0.35 , jobs = 1, rtrn = 0, dcct2_infos = ''):
    '''
    xy: an np array of 2 time series
    '''
    import numpy as np
    dst = {}
    for indx,no_of_zeroes in enumerate(zeros_list):
        # # # # print(indx, no_of_zeroes)

        p_val = 2*no_of_zeroes + 1

        x,y = augmented_zero_ts(x_original, y_original, no_of_zeroes = no_of_zeroes)



        # # # print(time.time())
        # # # t1 = time.time()


        dst[no_of_zeroes] = wping_paths_trnsfrmd(x.copy(), y.copy(), window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi= min(50,p_val),  p = p_val, m = 25, n = 10,  Threshold = 0.35 , jobs = 1, rtrn = 1)

        # dst[0]
        # dst[1]

        # # # # t2 = time.time() - t1
        # # # # print(t2/60)

        dst[no_of_zeroes].append(  best_path(dst[no_of_zeroes][1])  )
        dst[no_of_zeroes].append(  len( best_path(dst[no_of_zeroes][1]))  )

        dst[no_of_zeroes].append( create_lead_from_path( best_path(dst[no_of_zeroes][1]) , x,  no_of_zeroes )
                                )

    dst_df = pd.DataFrame.from_dict(dst).T

    dst_df.columns = ["d", "dtw", "vr" , "vc" , "mir", "mic", "mir_i", "no_terms_l", "no_terms_w", "path", 'length_path', 'lead']

    leadpdf = pd.DataFrame( np.vstack( dst_df.lead.values ) ).T

    leadpdf.columns = zeros_list

    leadpdf=leadpdf.fillna( method='ffill')
    leadpdf=leadpdf.fillna( method='bfill')


    (best_z, corr_z, leadpdf2) = best_z_val(x_original, y_original , leadpdf, zeros_list )

    # # # print(best_z)
    # # # print( leadpdf.columns)
    # # # print( leadpdf)
    
    lead = list( leadpdf[best_z].values)
    # # # print( leadpdf[best_z].values)
    
    # # printing functions here, get more parameters called
    
    dcct2_info_namer = dcct2_infos +  str( np.random.random())[2:8]
    import os
    try:
        os.mkdir(dcct2_info_namer )
    except FileExistsError:
        pass
    

    
    np.savetxt(dcct2_info_namer+'/x_original.out', x_original)
    np.savetxt(dcct2_info_namer+'/y_original.out', y_original)
    np.savetxt(dcct2_info_namer + '/lead.out', lead )
    leadpdf.to_csv(dcct2_info_namer + '/leadpdf')
    corr_z.to_csv(dcct2_info_namer + '/corr_z')
    
# # # # # # # # # # # # # # # # # # # # #     

    plots = [ lead ]
    pdf = pd.DataFrame(plots).T
    pdf.columns= [  'dcct2' ]
    
    pdf=pdf.fillna( method='ffill')
    pdf=pdf.fillna( method='bfill')
    
    fig1 = pdf.plot().get_figure()
    fig1.savefig(dcct2_info_namer + '/lead_fig')
    # fig1.savefig(dcct2_info_namer + '/lead_fig')
    
    plots = [x_original, y_original]
    pdf = pd.DataFrame(plots).T
    pdf.columns= [  'xR', 'yR' ]
    
    fig2 = pdf.plot().get_figure()
    fig2.savefig(dcct2_info_namer + '/xy_fig')
    
    plots = [np.cumsum(x_original), np.cumsum(y_original)]
    pdf = pd.DataFrame(plots).T
    pdf.columns= [  'xNPrice', 'yNPrice' ]
    
    fig3 = pdf.plot().get_figure()
    fig3.savefig(dcct2_info_namer + '/xyNPrice_fig')
    
    fig1.clf()
    fig2.clf()
    fig3.clf()
    
    mt.pyplot.close('all')
    
    
# # # # # # # # # # # # # # # # # # #     
    
    
    if rtrn == 0:
        # # # # print( corr_z.loc[best_z,'euclid_v'] )
        return corr_z.loc[best_z,'euclid_v']

    else:
        # # # # print( lead, leadpdf, corr_z )
        return lead, leadpdf, corr_z





# def get_lead_from_zero_wrapper(  ):















