import numpy as np 

import os
import datetime
import sklearn.preprocessing as prep
from sklearn import preprocessing

# rtscd = rtscd.values

import copy

import pandas as pd

from scipy.cluster import hierarchy as hr

import scipy

from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import fcluster

import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
import fastdtw

 

import scipy
import numpy  as np
import pandas as pd











def norming(x):
    return x / x[0]


def ssd_returns(X,Y , p = 100, m = 100, n = 18, Threshold = 0.35 , jobs = 1, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=100):
    '''
#This function returns the sum of squared differences between two lists, in addition the
#standard deviation of the spread between the two lists are calculated and reported.
    X = india_dt[:,0] Y = india_dt[:,1]
    '''
    def norming(x):
        if x[0] != 0:
            return x / x[0]
        else:
            return x
    
    X = norming(X.copy())
    Y = norming(Y.copy())
    spread = [] #Initialize variables
    std = 0
    cumdiff = 0
    for i in range(len(X)): #Calculate and store the sum of squares
        cumdiff += (X[i]-Y[i])**2
        spread.append(X[i]-Y[i])
    std = np.std(spread)  #Calculate the standard deviation
    return cumdiff


def ssd(X,Y , p = 100, m = 100, n = 18, Threshold = 0.35 , jobs = 1, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=100, dcct2_infos = ''):
    '''
#This function returns the sum of squared differences between two lists, in addition the
#standard deviation of the spread between the two lists are calculated and reported.
    X = india_dt[:,0] Y = india_dt[:,1]
    '''
    # def norming(x):
    #     if x[0] != 0:
    #         return x / x[0]
    #     else:
    #         return x
    
    
    X = np.cumsum( X )
    Y = np.cumsum( Y )
    
    
    # X = norming(X.copy())
    # Y = norming(Y.copy())
    spread = [] #Initialize variables
    std = 0
    cumdiff = 0
    for i in range(len(X)): #Calculate and store the sum of squares
        cumdiff += (X[i]-Y[i])**2
        spread.append(X[i]-Y[i])
    std = np.std(spread)  #Calculate the standard deviation
    return cumdiff

def get_distance_matrix(data , TYPE, simmilarity=0, load = 0 , load_name= None, I_TEMP = None, write_matrices= 0, write_name = '', P = 100, N = 13, M = 100, PSI = 100, dcct2_infos = ''):
    '''
    Should find the distance matrix or similarity matrix
    should be able to load the matrix by initial_name and i_temp
    load_name= 'dd_21nv_non_exp_dta_dtw_'
    '''
    if load == 1:
        dmatrix = pd.read_csv( load_name + str(I_TEMP) + '.txt' , sep = ' ', header=True , index = False )
        return dmatrix
    else:
        indian_dtadtw = pd.DataFrame( distance_matrix( data , p = P , m = M, n = N, type = TYPE , Threshold = 0.35 , jobs = 1, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=PSI , dcct2_infos = dcct2_infos) )
    

        if simmilarity == 0:
            dmatrix = np.log( (2/indian_dtadtw) - 1 )/4
        else:
            dmatrix = indian_dtadtw
            
        if write_matrices == 1:
            dmatrix.to_csv( 'dmatrix_' + str(I_TEMP) + write_name+ '.txt' , sep = ' ', header=True , index = False )
        
        return dmatrix




def find_sorted_pairs( matrix = None, simmilarity= 0  ):
    '''
    Change this paragraph with
    '''

    size = matrix.shape[0]
    
    col = np.array( [list(range(size))]*size )
    row = np.array( [list(range(size))]*size ).transpose()
    
    indd2 = pd.DataFrame( np.stack( ( matrix.values.reshape(1,-1)[0] , col.reshape(1,-1)[0] , row.reshape(1,-1)[0] )  ).transpose() , columns = [ 'dist', 'col' , 'row' ] )
    
    indd2.sort_values( by = 'dist' , inplace=True)
    
    lngth = len( indd2[size:].iloc[:,1:].values )
    double_removed = np.array( list( range(int( lngth/2) ) ) )*2
    
    dpairs = np.array( indd2[size:].iloc[:,1:].values[ double_removed ] , dtype = 'int')
    
    return dpairs



def find_sorted_pairs_vals( matrix = None, simmilarity= 0  ):
    '''
    Change this paragraph with
    '''

    size = matrix.shape[0]
    
    col = np.array( [list(range(size))]*size )
    row = np.array( [list(range(size))]*size ).transpose()
    
    indd2 = pd.DataFrame( np.stack( ( matrix.values.reshape(1,-1)[0] , col.reshape(1,-1)[0] , row.reshape(1,-1)[0] )  ).transpose() , columns = [ 'dist', 'col' , 'row' ] )
    
    indd2.sort_values( by = 'dist' , inplace=True)
    
    lngth = len( indd2[size:].iloc[:,1:].values )
    double_removed = np.array( list( range(int( lngth/2) ) ) )*2
    
    dpairs = np.array( indd2[size:].iloc[:,1:].values[ double_removed ] , dtype = 'int')
    
    return dpairs




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

    dist_tmp = np.sum( x_new_i*y_new_j )/( (np.sum( x_new_i**2 )**0.5)*(np.sum( y_new_j**2 )**0.5) ) 

    
    final_dist = 2/( 1 + np.e**(4*dist_tmp) )
    
    return final_dist


def warping_paths(s1, s2, window=None, max_dist=None,
                  max_step=None, max_length_diff=None, penalty=None, psi=None,  p = 100,    
                  m = 100, n = 18, Threshold = 0.35 , jobs = 1):
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
    """
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
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             
            # # # # d = (s1[i] - s2[j])**2
            d = inner_cdtw_distance( s1[i], s2[j] )
            # print(d)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             
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
            # return np.inf, dtw
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             
            return np.inf
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #             
    # print(dtw)
    dtw = np.sqrt(dtw)
    # print(dtw)
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
        else:
            dtw[ir, ic - psi + mic + 1:ic+1] = -1
            d = vc[mic]
    # # # # # # # # # # # # # # # # # # # # return d, dtw
    return d


def DTW_euclid(x, y, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=50,  p = 50, m = 100, n = 18,  Threshold = 0.35 , jobs = 1, rtrn = 0, dcct2_infos = ''):

    from dtaidistance import dtw
    # x = xy[:,0]
    # y = xy[:,1]
    d = dtw.distance(x, y, window=None, psi=None)
    # best_path = dtw.best_path(paths)
    # # dtwvis.plot_warpingpaths( x,  y, paths, best_path)
    # lead_E = create_lead_from_path(best_path, x, 0 )
    return d


    
    
    



def warping_paths_trnsfrmd(os1, os2, window=None, max_dist=None,
                  max_step=None, max_length_diff=None, penalty=None, psi=50,  p = 50, m = 100, n = 18,  Threshold = 0.35 , jobs = 1, rtrn = 0, dcct2_infos = ''):
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
    mir_i = 0 
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
        return (d, dtw, vr , vc , mir, mic, mir_i)





def discdtw( x=None ,y=None ,  p = 100, m = 100, n = 18, Threshold = 0.35 , jobs = 1, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=100):
    '''
    Args:
        n = number of chunks 
        p = chunk length
        m = lag value
        x = series 1
        y = series 2
    
    x = rtscd[:,0]
    y = rtscd[:,1]
    '''
    x_new = transform_ts( x , p=100 )
    y_new = transform_ts( y , p=100 )
    
    # dtwd = fastdtw.fastdtw( x=x_new, y=y_new , dist=cdtw_distance )
    dtwd = fastdtw.dtw( x=x_new, y=y_new , dist=inner_cdtw_distance )
    
    dtwd_msr = -dtwd[0]
    
    # print( dtwd_msr )
    
    return (dtwd_msr, dtwd[1])





def corr_dis( x , y ,  p = 100, m = 100, n = 18,  Threshold = 0.35 , jobs = 1, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=100, dcct2_infos = ''):
    '''
    Args:
        x = series 1
        y = series 2
    x = rtscd[:,0]
    y = rtscd[:,1]
    '''
    return (2*scipy.spatial.distance.correlation(  x , y ))**0.5


def cort_dis(x, y,  p = 100, m = 100, n = 18, Threshold = 0.35 , jobs = 1, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=100):
    '''
    Args:
        x = series 1
        y = series 2
    x = rtscd[:,0]
    y = rtscd[:,1]
    '''
    cort = np.sum( x*y )/( (np.sum(x**2))*(np.sum(y**2)) )**0.5
    modulated_cort = 2/(1 + np.e**(4*cort))
    snox = np.array( list(range( len(x) )) )
    x1 = np.stack(  (snox, x ) , axis = 1)
    snoy = np.array( list(range( len(y) )) )
    y1 = np.stack(  (snoy, y ) , axis = 1)
    dtw_dis = fastdtw.dtw( x1 , y1 , int(2) )
    final_dis = modulated_cort*dtw_dis[0]
    return final_dis


def disp( x=None ,y=None , n=13 ,p=100 ,m=100, Threshold = 0.35 , jobs = 1, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=100 , return_type = 0, dcct2_infos = '' ):
    '''
    Args:
        n = number of chunks 
        p = chunk length
        m = lag value
        x = series 1
        y = series 2
    
    x = rtscd[:,0]
    y = rtscd[:,1]
    
    disp(x ,y )
    '''
    # cormat = matrix(nrow = n,ncol = (2*m + 1))
    cormat = np.array( [[np.nan]*(n)]*(2*m + 1) )
    
    # lag = vector(length = n)
    lag = np.array( [np.nan]*(n) )
    
    # corr = vector(length = n)
    corr = np.array( [np.nan]*(n) )
    
    # inserting nan at 0 pos to make indexing same as in R
    x = np.insert( x , 0 , np.nan )
    y = np.insert( y , 0 , np.nan )
    
    # for(s in 1:n){
    for s in range(1, (n+1)):
        # for(i in 1:(2*m + 1)){
        for i in range( 1, (2*m + 1 + 1) ):
            # cormat[s,i] =  sum((x[((m+1)+(s-1)*p):((m+1)+(s)*p)]) * (y[((m+1)+(s-1)*p - m - 1 + i):((m+1)+(s)*p - m - 1 + i)])) / ( sqrt( sum((x[((m+1)+(s-1)*p):((m+1)+(s)*p)])^2) ) * sqrt( sum((y[((m+1)+(s-1)*p - m - 1 + i):((m+1)+(s)*p - m - 1 + i)])^2) ))
            
            d = np.sum( x[((m+1)+(s-1)*p):((m+1)+(s)*p)] * y[((m+1)+(s-1)*p - m - 1 + i):((m+1)+(s)*p - m - 1 + i)] ) / ( np.sqrt( np.sum(x[((m+1)+(s-1)*p):((m+1)+(s)*p)]**2) ) * np.sqrt( np.sum((y[((m+1)+(s-1)*p - m - 1 + i):((m+1)+(s)*p - m - 1 + i)])**2) ))
            if np.isnan(d):
                cormat[(i-1),(s-1)] = -1
            else:
                cormat[(i-1),(s-1)] = d 
    
        k = max(cormat[:,(s-1)]) 
        
        # #############################
        # for(i in 1:(2*m + 1)){ if(k == cormat[s,i]){ lag[s] = i -1 - m }  }
        for i in range((2*m + 1)):
            if(k == cormat[i,(s-1)]):
                lag[s-1] =  i -1 - m 
                
        # ############################
        corr[s-1] = k

    # index1 = lag[:]

    fstatis = sum(corr)/n
    assert not np.isnan(fstatis)
    # return(fstatis , index1)
    # return( fstatis , lag )
    
    if return_type == 1:
        return ( fstatis , corr, lag )
    else:
        return fstatis


def disnew( x=None ,y=None ,n=15 ,p=100 ,m=100 , Threshold = 0.35,   jobs = 1, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=100 ):
    '''
    Args:
        n = number of chunks 
        p = chunk length
        m = lag value
        x = series 1
        y = series 2
    
    x = rtscd[:,0]
    y = rtscd[:,1]
    '''

    # cormat = matrix(nrow = n,ncol = (2*m + 1))
    cormat = np.array( [[np.nan]*(n)]*(2*m + 1) )
    
    # lag = vector(length = n)
    lag = np.array( [np.nan]*(n) )
    
    # corr = vector(length = n)
    corr = np.array( [np.nan]*(n) )
    
    # inserting nan at 0 pos to make indexing same as in R
    x = np.insert( x , 0 , np.nan )
    y = np.insert( y , 0 , np.nan )
    
    # for(s in 1:n){
    for s in range(1, (n+1)):
        # for(i in 1:(2*m + 1)){
        for i in range( 1, (2*m + 1 + 1) ):
            # cormat[s,i] =  sum((x[((m+1)+(s-1)*p):((m+1)+(s)*p)]) * (y[((m+1)+(s-1)*p - m - 1 + i):((m+1)+(s)*p - m - 1 + i)])) / ( sqrt( sum((x[((m+1)+(s-1)*p):((m+1)+(s)*p)])^2) ) * sqrt( sum((y[((m+1)+(s-1)*p - m - 1 + i):((m+1)+(s)*p - m - 1 + i)])^2) ))
            
            d = np.sum( x[((m+1)+(s-1)*p):((m+1)+(s)*p)] * y[((m+1)+(s-1)*p - m - 1 + i):((m+1)+(s)*p - m - 1 + i)] ) / ( np.sqrt( np.sum(x[((m+1)+(s-1)*p):((m+1)+(s)*p)]**2) ) * np.sqrt( np.sum((y[((m+1)+(s-1)*p - m - 1 + i):((m+1)+(s)*p - m - 1 + i)])**2) ))
            if np.isnan(d):
                cormat[(i-1),(s-1)] = -1
            else:
                cormat[(i-1),(s-1)] = d 
    
        k = max(cormat[:,(s-1)]) 
        # # # for(i in 1:(2*m + 1)){ if(k == cormat[s,i]){ lag[s] = i -1 - m }  }
        # # for i in range((2*m + 1)):
        # #     if(k == cormat[(s-1),i]):
        # #         lag.append( i -1 - m )

        if k < Threshold:
            k = 0
        
        corr[s-1] = k

    # index1 = lag[:]

    fstatis = sum(corr)/n
    assert not np.isnan(fstatis)
    # return(fstatis , index1)
    return( fstatis )



def cluster_no( nm , clstrs):
    # nm = 1
    # clstrs = [7,8,2,2,5,1,1,3,4,8,3,5,5,6,8,7,3,4,6,5,7,6,7,8,6,4,8,5,3,1]
    tmp0 = np.array( np.array( clstrs ) == np.array( [nm]*len(clstrs) ) , dtype = 'float')
    tmp0[ tmp0 == 0 ] = np.nan
    return tmp0

def SimIJ( orig_ix , test_ix , orig , test ):
    # np.array( cluster_no( i, orig ) == cluster_no( i, orig ) , dtype = int).sum()
    i = orig_ix
    j = test_ix
    intrs = np.array( cluster_no( i, orig ) == cluster_no( j, test ) , dtype = int).sum()
    sumd = np.nan_to_num( cluster_no( i, orig ) ).sum() + np.nan_to_num( cluster_no( j, test ) ).sum()
    Simij = 2*intrs / sumd
    return Simij
    
    


def evaluation(orig , test):
    orig_max = np.max( orig )
    test_max = np.max( test )
    
    # SimIJ =  
    
    Sim_arr = []
    for i in range(1,orig_max+1): # i =1
    
        max_simij = -100
        for j in range(1,test_max+1): # j =1
            Simij = SimIJ( i, j , orig , test )
            if max_simij < Simij:
                max_simij = Simij
        
        Sim_arr.append( max_simij )
        
    return np.mean( Sim_arr )




def distance_matrix(rtscd , p = 100, m = 100, n = 18, type = None, Threshold = 0.35 , jobs = 1, window=None, max_dist=None, max_step=None, max_length_diff=None, penalty=None, psi=100, dcct2_infos = ''):
    import sklearn.metrics as sk_metrics
    lan = []
    # n=6 ,p=120 ,m=20
    # print( '{} {}'.format(i,j) )

    # # # dd = 1 - d2
    kwds = {'n':n, 'p':p, 'm':m, 'Threshold':Threshold ,
            'window':window,
            'max_dist':max_dist,
            'max_step':max_step,
            'max_length_diff':max_length_diff,
            'penalty':penalty,
            'psi':psi,
            'dcct2_infos':dcct2_infos 
            }
    
    X = rtscd.transpose()
    d2 = sk_metrics.pairwise_distances( X , metric= type , n_jobs= jobs, **kwds )
    
    # if type == 'cdtw' or type == 'dta_dtw':
    #     return 2000 - d2
    
    if type == ssd:
        return d2
    
# # # # # # # # # # # # # # # # # # # # # # # # # # 
# get this thing below right 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
    x=4
    dd = 2/(1 + np.exp(x*d2))
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
    
    return dd





def evaluate_msr( dd, djia_cluster , namer = '' , lbls = None , method_linkage = 'ward' ):
    condensed_distance_matrix = scipy.spatial.distance.squareform( dd, force = 'tovector', checks = False )
    
    distance_matrix = scipy.spatial.distance.squareform( condensed_distance_matrix, force = 'tomatrix', checks = False )
    
    # Z = hr.linkage(condensed_distance_matrix , method= 'single' , optimal_ordering = True )  # form tree
    
    Z = hr.linkage( condensed_distance_matrix , method= method_linkage )  # form tree
    
    
    # Z = hr.linkage(m_l1 , method= 'complete', metric = distant_half_indiv )  # form tree
    
    # calculate full dendrogram# calcul 
    # # # # mt.pyplot.figure(figsize=(25, 10))
    # # # # mt.pyplot.title('Hierarchical Clustering Dendrogram' + str( np.random.randint(100,999) ) + namer)
    # # # # mt.pyplot.xlabel('sample index')
    # # # # mt.pyplot.ylabel('distance')
    # # # # dendrogram(
    # # # #     Z,
    # # # #     leaf_rotation=90.,  # rotates the x axis labels
    # # # #     leaf_font_size=8.,  # font size for the x axis labels
    # # # #     labels = lbls
    # # # # )
    # plt.show()
    # plt.savefig( './/results_table//' )
    # plt.close('all')
    
    # djia_cluster = [7,8,2,2,5,1,1,3,4,8,3,5,5,6,8,7,3,4,6,5,7,6,7,8,6,4,8,5,3,1]
    
    msrs = []
    
    for max_c in range(2, 25):
        # max_c = 8
        clusters = fcluster(Z, max_c, criterion='maxclust')
        clusters
        
        # #####################################################################
        
        # # # # # # # CLUSTER EVALUATION ###################################
        
        clusters, djia_cluster
        
        orig = djia_cluster.copy()
        test = clusters.copy()
        
        msr = evaluation(orig , test)
        
        msrs.append( [max_c, msr] )
        
        # print( msr )
    fmsrs = pd.DataFrame( np.array( msrs ) , columns = ['cluster', 'measure'] )
    
    return fmsrs, None




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def proper_cluster_name( ground_truth  ):
    # ground_truth = [7,8,2,2,5,1,1,3,4,8,3,5,5,6,8,7,3,4,6,5,7,6,7,8,6,4,8,5,3,1]

    starr = np.stack( ( list( range( len(ground_truth) ) ), ground_truth ), axis = 1 )
    starrpd = pd.DataFrame( starr , columns = ['ix', 'cl'])
    
    grpd = starrpd.groupby( by = 'cl' )
    
    ifnl = []
    
    for k,i in grpd:
        # print(i,k,len(i))
        # len(i)  
        i['ad'] = range( 1, 1 + len(i)   )
        i1 = pd.DataFrame( np.array( i.values , dtype = 'str' ) , columns = i.columns )
        i1['nm'] = i1['cl'] + '.' + i1['ad']
        ifnl.append( i1 )
    
    ifnl1 = pd.concat( ifnl )
    ifnl1 = pd.DataFrame( np.array(ifnl1.values , dtype = 'float') , columns = ifnl1.columns )
    ifnl2 = ifnl1.sort_values( by = 'ix' )
    
    return np.stack( (ground_truth, ifnl2.nm.values.tolist() ), axis = 0 )


# #################################################################



def multi_obj_pairs( dpairs0 = None , dpframe = None ):
    '''
    Input:
        dpairs0
        dpairs1
        type
        type_parameters

    Output:
        dpairs

    pygmo will return the index of best pairs , which can then be picked from dpairs
    first find non-dominated fronts and then sort by second 'TYPE2'
    '''
    import pygmo as pg
    import numpy as np
    import pandas as pd

    yt = dpframe.iloc[:, 2: ].values

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting( points = np.array( yt , dtype = 'float64') )

    ndf2 = []
    for frnt in ndf:
        # frnt = ndf[0]
        tmp_frnt = np.argsort( yt[:,1][ frnt ]  )
        ndf2.append( frnt[tmp_frnt] )

    sf = np.concatenate( ndf2 )
    # Invest equally in it
    fn_dpairs = dpairs0[ sf ]

    return fn_dpairs


def find_sorted_pairs_vals( matrix = None, simmilarity= 0  ):
    '''
    Change this paragraph with
    matrix = indd1_new0

    These values need to be minimised as required by pygmo

    '''
    size = matrix.shape[0]

    col = np.array( [list(range(size))]*size )
    row = np.array( [list(range(size))]*size ).transpose()

    indd2 = pd.DataFrame( np.stack( ( matrix.values.reshape(1,-1)[0] , col.reshape(1,-1)[0] , row.reshape(1,-1)[0] )  ).transpose() , columns = [ 'dist', 'col' , 'row' ] )

    indd2.sort_values( by = 'dist' , inplace=True)

    lngth = len( indd2[size:].iloc[:,1:].values )
    double_removed = np.array( list( range(int( lngth/2) ) ) )*2

    dpairs = np.array( indd2[size:].iloc[:,1:].values[ double_removed ] , dtype = 'int')
    dpair_values =  indd2[size:].iloc[:,0].values[ double_removed ]

    return dpair_values




