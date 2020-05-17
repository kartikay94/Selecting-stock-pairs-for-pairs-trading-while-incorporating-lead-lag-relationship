
import pandas as pd
import numpy as np
from disp000 import *

from dcct2_functions import *
import numpy as np

from sklearn import preprocessing
import argparse

import matplotlib as mt
mt.use('Agg')
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np



parser = argparse.ArgumentParser(description='Clustering')
parser.add_argument("--jobs", type=int, default=1)
parser.add_argument("--Threshold", type=float, default=0.35)
parser.add_argument("--fnm", type=str, default='fnm_')
parser.add_argument("--data_name", type=str, default='topix')
parser.add_argument("--sorted_pairs_simmilarity1", type=int, default=0)
parser.add_argument("--sorted_pairs_simmilarity2", type=int, default=1)
parser.add_argument("--roll_s", type=int, default=0)
parser.add_argument("--roll_e", type=int, default=6)

parser.add_argument("--P", type=int, default=100)
parser.add_argument("--M", type=int, default=100)
parser.add_argument("--N", type=int, default=3)
parser.add_argument("--PSI", type=int, default=100)
parser.add_argument("--period", type=int, default=21)

parser.add_argument("--strategy_no", type=int, default=8)
FUNCTION_MAP1 = {'disp' : disp,
                'dta_dtw': warping_paths_trnsfrmd,
                'corr': corr_dis,
                'ssd': ssd,
                'dcct2': get_lead_from_zero,
                'dtw_e': DTW_euclid
                }

parser.add_argument('--dist_func1', choices=FUNCTION_MAP1.keys(), default= 'corr')

parser.add_argument('--dist_func2', choices=FUNCTION_MAP1.keys(), default= 'ssd')

args = parser.parse_args()

TYPE1 = FUNCTION_MAP1[ args.dist_func1 ]
TYPE2 = FUNCTION_MAP1[ args.dist_func2 ]



write_name = 'rslt_'+ args.data_name +'_' + args.dist_func1 + '_psi' + str(args.PSI) + '_p' + str(args.P) + '_' + str( np.random.random())[2:6] + args.fnm

new_dir = 'rslts_march2020' + args.data_name + '_period' + str(args.period)

    
import os

try:
    os.mkdir(new_dir)
except FileExistsError:
    pass

try:
    os.mkdir(new_dir + '/result')
except FileExistsError:
    pass
    
try:
    os.mkdir(new_dir + '/dcct2_info')
except FileExistsError:
    pass

try:
    os.mkdir(new_dir + '/arg')
except FileExistsError:
    pass


names_tmp1 = pd.read_csv( 'backtradert_pairs//' + args.data_name + "_price.csv", index_col='date' ).columns[0]

aapl = pd.read_csv( 'backtradert_pairs//'+args.data_name +'//'+names_tmp1+'.csv' )




cashes_i_tmp = []                 # ##############
# # # # # # # # # # # # # # # # # # # # 
for i_tmp in range(args.roll_s, args.roll_e):              # ############## i_tmp = 0
    dcct2_info_namer = new_dir + '/dcct2_info' + '/' + write_name + '_'+str(i_tmp) +'_'
    

    data = indian_data(data_name = args.data_name)[0][ (250*i_tmp):(500 + 250*i_tmp) , :]
    

    indd1_new0 = get_distance_matrix( data , TYPE1, simmilarity= args.sorted_pairs_simmilarity1 , load = 0 , load_name= None, I_TEMP = None, write_matrices = 0, write_name = '', P = args.P, M = args.M, N = args.N, PSI = args.PSI , dcct2_infos = dcct2_info_namer )
    
    dcct2_info_namer + '_DM'
    indd1_new0.to_csv( dcct2_info_namer + '_DM_'+ args.dist_func1 + '.txt' , sep = ' ', header=True , index = False )
    
    indd1_new1 = get_distance_matrix( data , TYPE2, simmilarity= args.sorted_pairs_simmilarity2 , load = 0 , load_name= None, I_TEMP = None, write_matrices = 0, write_name = '', P = args.P, M = args.M, N = args.N, PSI = args.PSI )
    
    indd1_new1.to_csv( dcct2_info_namer + '_DM_'+ args.dist_func2 + '.txt' , sep = ' ', header=True , index = False )


    dpairs0 =  np.sort( find_sorted_pairs( matrix = indd1_new0, simmilarity= 0 ), axis=1 )

    dpairs1 =  np.sort( find_sorted_pairs( matrix = indd1_new1, simmilarity= 0  ), axis=1 )

    dpairs0_vals =  find_sorted_pairs_vals( matrix = indd1_new0, simmilarity= 0  )

    dpairs1_vals =  find_sorted_pairs_vals( matrix = indd1_new1, simmilarity= 0  )


    dp_vals0 = pd.DataFrame( np.stack( (  dpairs0[:,0], dpairs0[:,1] , dpairs0_vals ) ).T , columns = ['dp0_0', 'dp0_1' , 'dp0_val' ] )
    dp_vals0['indx'] = dp_vals0['dp0_0']*100 + dp_vals0['dp0_1']

    dp_vals1 = pd.DataFrame( np.stack( ( dpairs1[:,0], dpairs1[:,1] , dpairs1_vals ) ).T , columns = ['dp0_0', 'dp0_1', 'dp1_val' ] ) # ['dp1_0', 'dp1_1', 'dp1_val' ]
    dp_vals1['indx'] = dp_vals1['dp0_0']*100 + dp_vals1['dp0_1']

    dp_vals_comb = dp_vals0.merge(dp_vals1.loc[:,[ 'indx' , 'dp1_val' ]], on= [ 'indx' ] )

    dp_vals_comb.drop( ['indx'], axis=1, inplace=True )


    dpairs_new = multi_obj_pairs( dpairs0 = dpairs0 , dpframe = dp_vals_comb )



    djia_names1 = pd.read_csv( 'backtradert_pairs//' + args.data_name + "_price.csv", index_col='date' ).columns

    # djia_names1 = ['HDFC', 'ICICI', 'AXIS', 'KOTAK', 'INDUSIND', 'Yes_Bank_Ltd','Federal_Bank_Ltd', 'Karur_Vysya_', 'South_Indian_B_L', 'INDIAN_OIL','ONGC', 'BHARAT', 'ESSAR', 'Cairn_I_L', 'Hindustan_P_C_','Aban_Offshore', 'Hind_Oil_exp', 'Manga_Refinery', 'Sbi_Bank', 'Baroda','PNB', 'IDBI_Ltd', 'Cent_Ban_I', 'Canara_Bnk', 'UBI', 'Bank_Of_India','Syndicate', 'Indian_Bank']


    imprtfl = 'backtradert_pairs.pair_trade' + str( args.strategy_no )
    imprtfl2 = 'from ' + imprtfl + ' import *'
    exec( str(imprtfl2) )


    cashes = []

    ll_count  = -1
    max_ll_count = 30

    for i in range( 500 ):
        ll_count += 1
        if ll_count > max_ll_count:
            break


        p1 = dpairs_new[i]

        initial_names = 'backtradert_pairs//' + args.data_name + '//'

        p1d1 = initial_names + djia_names1[ p1[0] ] + '.csv'
        p1d2 = initial_names + djia_names1[ p1[1] ] + '.csv'

        fdte = aapl.Date.values[ (500 + 250*i_tmp) ]          # ##############
        tdte = aapl.Date.values[ (750 + 250*i_tmp) ]          # ##############

        tcash = runstrategy( args_data0 = p1d1 , args_data1 = p1d2 , args_fromdate = fdte , args_todate =  tdte , period = args.period  , oldsync=False )                                         # ##############

        cashes.append( tcash )

    cashes_i_tmp.append( cashes )                             # ##############

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    print( cashes )
    print( np.mean( cashes ) )
    print('dta_dtw')

print( 'cashes_i_tmp' )
print( cashes_i_tmp )

cashes_i_tmp = np.array( cashes_i_tmp )
cashes_i_tmp.mean( axis = 1 )
cashes_i_tmp.mean( axis = 1 ).mean()


pd.DataFrame( cashes_i_tmp ).to_csv(  new_dir + '/result' +'/'+write_name + '.csv' )
pd.DataFrame( [str(args)] ).to_csv( new_dir + '/arg' +'/'+ 'args' + write_name + '.csv' )





