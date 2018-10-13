
# coding: utf-8

# Usage:
# python3 xxxx.py [cv_fold_nr] [anomaly_method_int] [score_method_int] [searchable_method_int] [top_tuple_method_int]

# In[5]:


# custom lib
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
from joblib import Parallel, delayed

# build-in lib
import time
from datetime import datetime
import math
import sys


# In[6]:


if('ipykernel_launcher' in sys.argv[0]):
    using_jupyter_gui = True
else:
    using_jupyter_gui = False


# In[7]:


debugging = True
use_spark = False # CPU joblib is fast enough, 1 hour for each fold (8 cores). Spark may give java heap outOfMemory error.
print_verbose_info = False
save_verbose_results = False
use_cache = True
save_predictions = True
Random_Seed_Default = 0
cv_random_seed = Random_Seed_Default

# automatic param
if(not using_jupyter_gui):
    debugging = False
if(using_jupyter_gui):
    print_verbose_info = True
if(use_cache):
    cv_random_seed = 0 # 0 !!!
    cache_file_folder = '/root/Dropbox/detect-r-proj/cache/'
    py_hexversion = sys.hexversion
    cache_file_prefix = cache_file_folder + 'v13_e2d_' + str(py_hexversion) + '_' + str(pd.__version__) + '_' + str(sklearn.__version__) + '_'


# In[8]:


import warnings
import itertools
#import statsmodels.api as sm

if(using_jupyter_gui):
    import matplotlib.pyplot as plt
    #plt.style.use('ggplot')
    plt.style.use('fivethirtyeight')


# ### algorithm parameters

# In[9]:


make_min_score_zero = True # WPT .cu code is also doing so in normalizeThresholdTreatment()
measurement_metric = 'MAE' # MAE. Can use MAPE, etc., but need to modify GPU raw results and add more results to results_of_all_folds_df

mid_size_file_storage_path = 'log/'
gpu_results_binary_file_path = '/dev/shm/' # DO USE shm for efficiency, 5 times faster than NFS.
big_file_storage_path = '/mnt/nfsMountPoint/datasets/detect_cluster/'
missing_flag_binary_npy_file = '/root/Dropbox/detect-r-proj/incident_flag_binary.missingOnly.npy'
anomaly_flag_binary_npy_file = None
top_tuple_percentage = None

# set some defaults, will be over-written by CLI
cv_fold_nr = 3
anomaly_method_int = 0
score_method_int = 0
searchable_method_int = 0
top_tuple_method_int = 0

if(not using_jupyter_gui): # CLI command line
    if(len(sys.argv) > 1):
        cv_fold_nr = int(sys.argv[1]) # 2~10
    if(len(sys.argv) > 2):
        anomaly_method_int = int(sys.argv[2]) # 0:DecompositionWeek, 1:S-H-ESD, 2:true_incident_label, 3:true_accident_label
    if(len(sys.argv) > 3):
        score_method_int = int(sys.argv[3]) # 0:subtraction_linear, 1:reciprocal... ref: compareScoreMethod.png
    if(len(sys.argv) > 4):
        searchable_method_int = int(sys.argv[4]) # 0: incident & normal both as neighbours, 1: separate them.
    if(len(sys.argv) > 5):
        top_tuple_method_int = int(sys.argv[5]) # 0:top25%, 1:top1, 2:all, 3:top3, 4:top50%, 10:top10, 6:top~X% (for nex param???).
    if(len(sys.argv) > 6):
        top_tuple_percentage = float(sys.argv[6])

# automatic params:

if(cv_fold_nr < 2 or cv_fold_nr > 10):
    print('Unknown cv_fold_nr' + str(cv_fold_nr))
    quit()

if(anomaly_method_int == 0):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-gsw-missingOnly.bin'
    anomaly_method = 'Decomposition_Week_ResNormAvg'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_DecompositionWeek_residual_norm_avg.npy' # '/root/Dropbox/detect-r-proj/anomaly_flag_406_binary.npy'
elif(anomaly_method_int == 1):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-gsw-missingOnly.bin'
    anomaly_method = 'Decomposition_Week_Pure'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_DecompositionWeek_Pure.npy'
elif(anomaly_method_int == 2):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-gsw-missingOnly.bin'
    anomaly_method = 'Ground_Truth_Incident'
    anomaly_flag_binary_npy_file = 'incidentFlagBinary.npy'
elif(anomaly_method_int == 3):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-gsw-missingOnly.bin'
    anomaly_method = 'Ground_Truth_Accident'
    anomaly_flag_binary_npy_file = 'incidentFlag_digits013_NonAccidentIncidentAs_0.npy'
elif(anomaly_method_int == 4):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-gsw-missingOnly.bin'
    anomaly_method = 'S_H_ESD_ResNormAvg'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_S_H_ESD_residual_norm_avg.npy'
elif(anomaly_method_int == 5):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-gsw-missingOnly.bin'
    anomaly_method = 'S_H_ESD_Pure'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_S_H_ESD_Pure.npy'
elif(anomaly_method_int == 6):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-missingAndDecompositionWeek_ResNormAvg.bin'
    anomaly_method = 'DecompositionWeek_ResNormAvg_NN_Aware'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_DecompositionWeek_residual_norm_avg.npy'
elif(anomaly_method_int == 7):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-missingAndDecompositionWeek_ResNormAvg_reversed.bin'
    anomaly_method = 'DecompositionWeek_ResNormAvg_NN_Aware_Reversed'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_DecompositionWeek_residual_norm_avg.npy'
elif(anomaly_method_int == 8):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-missingAndS_H_ESD_ResNormAvg.bin'
    anomaly_method = 'S_H_ESD_ResNormAvg_NN_Aware'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_S_H_ESD_residual_norm_avg.npy'
elif(anomaly_method_int == 9):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-missingAndS_H_ESD_ResNormAvg_reversed.bin'
    anomaly_method = 'S_H_ESD_ResNormAvg_NN_Aware_Reversed'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_S_H_ESD_residual_norm_avg.npy'
elif(anomaly_method_int == 10):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-missingAndDecompositionWeek_ResNormAvg_combined.bin'
    anomaly_method = 'DecompositionWeek_ResNormAvg_NN_Aware_combined'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_DecompositionWeek_residual_norm_avg.npy'
elif(anomaly_method_int == 11):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-missingAndS_H_ESD_ResNormAvg_combined.bin'
    anomaly_method = 'S_H_ESD_ResNormAvg_NN_Aware_combined'
    anomaly_flag_binary_npy_file = 'anomalyFlag406_S_H_ESD_residual_norm_avg.npy'
elif(anomaly_method_int == 12):
    gpu_results_binary_file = gpu_results_binary_file_path + 't145325_expResult-v13.1-detect-gsw-missingOnly.bin'
    anomaly_method = 'Ground_Truth_Incident'
    anomaly_flag_binary_npy_file = 'incidentFlag_digits013_missingAs3.npy'
else:
    print('Unknown anomaly_method_int' + str(anomaly_method_int))
    quit()

if(score_method_int == 0):
    score_method = 'Subtraction_Linear'
elif(score_method_int == 1):
    score_method = 'Reciprocal_Square'
elif(score_method_int == 2):
    score_method = 'Exponential_Dist'
else:
    print('Unknown score_method_int' + str(score_method_int))
    quit()

if(searchable_method_int == 0):
    searchable_method = 'Both_Neighbours'
elif(searchable_method_int == 1):
    print('Unknown searchable_method_int yet:' + str(score_method_int))
    quit()
    searchable_method = 'Separate_Neighbours'
else:
    print('Unknown searchable_method_int' + str(score_method_int))
    quit()

if(top_tuple_method_int == 0):
    top_tuple_method = 'Top25Percent'
elif(top_tuple_method_int == 1):
    top_tuple_method = 'Top1'
elif(top_tuple_method_int == 3):
    top_tuple_method = 'Top3'
elif(top_tuple_method_int == 10):
    top_tuple_method = 'Top10'
elif(top_tuple_method_int == 2):
    top_tuple_method = 'All'
elif(top_tuple_method_int == 4):
    top_tuple_method = 'Top50Percent'
elif(top_tuple_method_int == 6):
    top_tuple_method = 'TopXPercent'
else:
    print('Unknown top_tuple_method_int' + str(anomaly_method_int))
    quit()

if(top_tuple_method == 'TopXPercent' and (top_tuple_percentage < 0 or top_tuple_percentage > 1)):
    print('Unknown top_tuple_percentage' + str(top_tuple_percentage))
    quit()

cache_config_string = 'config_' + str(cv_fold_nr) + '_' + str(anomaly_method_int) + '_' + str(score_method_int) + '_' + str(searchable_method_int)
current_config_string = cache_config_string + '_' + str(top_tuple_method_int)


# In[10]:


print('cv_fold_nr: ' + str(cv_fold_nr))
print('anomaly_method: ' + str(anomaly_method))
print('score_method: ' + str(score_method))
print('searchable_method: ' + str(searchable_method))
print('top_tuple_method: ' + str(top_tuple_method))
print('make_min_score_zero: ' + str(make_min_score_zero))
print('measurement_metric: ' + str(measurement_metric))


# In[11]:


complete_day_nr = 406
daily_record_nr = 288
data_variable_nr = 2
search_step_length_listing  = [ 2, 4, 8, 16, 32, 64, 128, 256 ]
window_size_listing         = [ 0, 4, 8, 16, 32 ]
k_level1_listing            = [ 2, 4, 8, 16, 32, 64, 128, 256 ]
predict_step_length_listing = [ 1, 2, 4, 8 ]

combination_nr_per_time_point_no_prediction = len(search_step_length_listing) * len(window_size_listing) * len(k_level1_listing)
combination_nr_per_time_point = data_variable_nr * len(predict_step_length_listing) * combination_nr_per_time_point_no_prediction
print('combination_nr_per_time_point_no_prediction: ' + str(combination_nr_per_time_point_no_prediction))


# In[12]:


exp_started_datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")
#exp_started_datetime_string = datetime.now().strftime("%Y%m%d_%H%M")


# In[13]:


filename_prefix = 'started_GMT_' + exp_started_datetime_string
if(debugging):
    filename_prefix = 'debugging-' + filename_prefix
filename_data = filename_prefix + '.npy'
param_string = '_param_' + '_'.join(sys.argv[1:])
filename_log = 'log/' + filename_prefix + param_string + '.log.txt'
print(filename_log)


# In[14]:


if(not using_jupyter_gui and use_spark):
    from pyspark import SparkContext, SparkConf
    appName = 'detect'
    master = 'spark://sparkmaster.dmml.stream:7077'
    conf = SparkConf().setAppName(appName + exp_started_datetime_string).setMaster(master)
    sc = SparkContext(conf=conf)


# In[15]:


# redirect output print to a log file
if(not using_jupyter_gui):
    import sys
    orig_stdout = sys.stdout
    f_stdout = open(filename_log, 'w', 2) # 0:non-buffer, only for bin. 1:line. 2:2bytes.
    sys.stdout = f_stdout
    
    print(filename_log)
    print('debugging: ' + str(debugging))
    print('use_spark: ' + str(use_spark))


# In[16]:


import inspect, os
print(inspect.getfile(inspect.currentframe())) # script filename (usually with path)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) # script directory
print(os.path.abspath(inspect.stack()[0][1]))
print('current_config_string: ' + current_config_string)


# # Read Data

# In[17]:


def status_measurement_label_listing(one_point_index):
    label_listing = [0, 0, 0, 1, 0]
    return(label_listing)


# if(use_spark):
#     ansibleslaves 'if not copied: cp /root/nfsMountPoint/datasets/t145325_expResult-v11.1-detect-gsw-missingOnly.bin /dev/shm/'

# In[18]:


# remove prefix, suffix(trailing).
global_index_truncated = list(range((daily_record_nr + max(predict_step_length_listing)), complete_day_nr * daily_record_nr))
missing_flag_binary = np.load(missing_flag_binary_npy_file)
if(len(missing_flag_binary) < complete_day_nr * daily_record_nr):
    print('ERR: len(missing_flag_binary) < complete_day_nr * daily_record_nr')
    quit()
anomaly_flag_binary = np.load(anomaly_flag_binary_npy_file)
if(len(anomaly_flag_binary) < complete_day_nr * daily_record_nr):
    print('ERR: len(anomaly_flag_binary) < complete_day_nr * daily_record_nr')
    quit()
missing_flag_binary_truncated = missing_flag_binary[(daily_record_nr + max(predict_step_length_listing)):(complete_day_nr * daily_record_nr)]
anomaly_flag_binary_truncated = anomaly_flag_binary[(daily_record_nr + max(predict_step_length_listing)):(complete_day_nr * daily_record_nr)]

# set missing for stratifiled k folds
anomaly_flag_binary_truncated[missing_flag_binary_truncated == 1] = 3

# the last complete day (406, index 405) is not analyzed. // but searchable?
missing_flag_binary[(complete_day_nr - 1) * daily_record_nr : -1] = 1


# In[19]:


global_index_and_label_truncated_df = pd.DataFrame(
    {'global_index': global_index_truncated,
     'label': anomaly_flag_binary_truncated
    }
)


# In[20]:


global_index_listing = np.array(range(0, complete_day_nr * daily_record_nr))
global_index_listing


# In[21]:


missing_value_global_index_listing = np.where(missing_flag_binary == 1)
missing_value_global_index_listing


# global_index_multiply_config_nr_listing = [i 
#                                                        for i in global_index_listing
#                                                        for j in range(combination_nr_per_time_point_no_prediction)]
# print(global_index_multiply_config_nr_listing[319], global_index_multiply_config_nr_listing[320])

# global_index_non_missing_listing = global_index_listing[np.logical_not(np.isin(global_index_listing, missing_value_global_index_listing))]
# len(global_index_non_missing_listing)
# global_index_non_missing_multiply_config_nr_listing = [i 
#                                                        for i in global_index_non_missing_listing
#                                                        for j in range(combination_nr_per_time_point_no_prediction)]
# print(global_index_non_missing_multiply_config_nr_listing[319], global_index_non_missing_multiply_config_nr_listing[320])

# ### config some env variables

# In[22]:


### for debugging
partition_nr = 10 #sc.defaultParallelism


# In[23]:


# init both CPU and Spark for debugging convenience, no big influence for deployment.

# parallel CPU threads.
from joblib import Parallel, delayed
import os
def get_nr_cpu_threads():
    #for Linux, Unix and MacOS
    if hasattr(os, "sysconf"):
        if 'SC_NPROCESSORS_ONLN' in os.sysconf_names:
            #Linux and Unix
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:
            #MacOS X
            return int(os.popen2("sysctl -n hw.ncpu")[1].read())
    #for Windows
    if 'NUMBER_OF_PROCESSORS' in os.environ:
        ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
        if ncpus > 0:
            return ncpus
    #return the default value
    return 1

print("joblib cpu_threads_nr: %d" % get_nr_cpu_threads())
parallel_job_nr = get_nr_cpu_threads() - 1

# spark RDD partitions.
if('sc' in vars()):
    if(debugging):
        partition_nr = 10 #sc.defaultParallelism
    else:
        partition_nr = min(int(3.5 * sc.defaultParallelism), 1000)
    print("sc.defaultParallelism (cores): %d" % sc.defaultParallelism)
    print("RDD partition_nr: %d" % partition_nr)


# # Training

# ## config some fixed algorithm variables

# In[24]:


measurement_column_listing = ['err_flow1', 'err_speed1', 'err_flow2', 'err_speed2', 'err_flow4', 'err_speed4', 'err_flow8', 'err_speed8']

# automatic:
score_column_listing = [i.replace('err_', 'score_') for i in measurement_column_listing]
score_result_column_listing = score_column_listing.copy()
score_result_column_listing.append('config_index')
weight_column_listing = [i.replace('err_', 'weight_') for i in measurement_column_listing]
weight_result_column_listing = weight_column_listing.copy()
weight_result_column_listing.append('config_index')


# ## functions for non-labelled & labelled

# if not copied
# ansibleslaves '[ -e /dev/shm/t145325_expResult-v11.1-detect-gsw-missingOnly.bin ] && : || cp /root/nfsMountPoint/datasets/t145325_expResult-v11.1-detect-gsw-missingOnly.bin /dev/shm/'

# In[25]:


item_size_bytes = 4 # 4: 32bits, 8: 64bit
file_input_name = gpu_results_binary_file


# In[26]:


def train_one_point(one_train_index):
    seek_nr = one_train_index * combination_nr_per_time_point * item_size_bytes
    p_file = open(file_input_name, "rb")
    p_file.seek(seek_nr, os.SEEK_SET)
    t = np.fromfile(p_file, dtype=np.float32, count=combination_nr_per_time_point)
    t = pd.DataFrame(t.reshape((combination_nr_per_time_point_no_prediction, len(predict_step_length_listing) * data_variable_nr))) # 320 * 8
    t.columns = measurement_column_listing
    t['config_index'] = range(0, combination_nr_per_time_point_no_prediction) # 0, 320
    
    #t145325_expResult_train_selected_df = t145325_expResult_df[np.isin(global_index_multiply_config_nr_listing, one_train_index)] # to-improve: use exact index directly
    #t = t145325_expResult_train_selected_df.groupby(['config_index']).mean().abs().reset_index().join(config_indexed_df.set_index('config_index'), on='config_index')
    
    if(measurement_metric != 'MAE'):
        stop('ERR: unexpected measurement_metric')
    for measurement_column in measurement_column_listing:
        t.sort_values(measurement_column, inplace=True)
        score_column_name = measurement_column.replace('err_', 'score_')
        t[score_column_name] = range(319, -1, -1)
    t = t[score_result_column_listing]
    t.sort_values('config_index', inplace=True)
    return(t)


# In[27]:


def get_train_all_points_all_tuples_scores_df(global_index_train_listing):
    if(debugging):
        from timeit import default_timer as timer
        start = timer()
        if(not use_spark):
            all_points_all_tuples_result_listing = Parallel(n_jobs=parallel_job_nr)(delayed(train_one_point)(one_train_index) for one_train_index in global_index_train_listing[:30])
        else:
            my_rdd = sc.parallelize(list(global_index_train_listing[:10]))
            map_result = my_rdd.map(train_one_point)
            all_points_all_tuples_result_listing = map_result.collect()
        end = timer()
        print('use_spark: ' + str(use_spark) + '. time used: ' + str(end - start) + 's.')
    else:
        if(not use_spark):
            all_points_all_tuples_result_listing = Parallel(n_jobs=parallel_job_nr)(delayed(train_one_point)(one_train_index) for one_train_index in global_index_train_listing)
        else:
            my_rdd = sc.parallelize(list(global_index_train_listing), partition_nr)
            map_result = my_rdd.map(train_one_point)
            all_points_all_tuples_result_listing = map_result.collect()
    all_points_all_tuples_scores_df = pd.concat(all_points_all_tuples_result_listing).groupby('config_index').sum().reset_index()
    return(all_points_all_tuples_scores_df)


# In[28]:


def get_scores_weights_from_scores(all_points_all_tuples_scores_df):
    for one_score_column in score_column_listing:
        all_points_all_tuples_scores_df.sort_values(one_score_column, ascending=False, inplace=True)

        # fine tuning of weights: check WPT to make sure which step comes first: make_min_score_zero vs. discard_some_bad_tuples
        if(make_min_score_zero):
            all_points_all_tuples_scores_df[one_score_column] -= all_points_all_tuples_scores_df[one_score_column].min()
        all_points_all_tuples_scores_df = all_points_all_tuples_scores_df.reset_index(drop=True)
        if(top_tuple_method == 'Top25Percent'):
            top_tuple_percentage = 0.25
            top_tuple_nr = int(all_points_all_tuples_scores_df.shape[0] * top_tuple_percentage)
            all_points_all_tuples_scores_df.loc[top_tuple_nr:, [one_score_column]] = 0
        elif(top_tuple_method == 'Top50Percent'):
            top_tuple_percentage = 0.50
            top_tuple_nr = int(all_points_all_tuples_scores_df.shape[0] * top_tuple_percentage)
            all_points_all_tuples_scores_df.loc[top_tuple_nr:, [one_score_column]] = 0
        elif(top_tuple_method == 'All'):
            pass
        elif(top_tuple_method == 'Top1'):
            top_tuple_nr = 1
            all_points_all_tuples_scores_df.loc[top_tuple_nr:, [one_score_column]] = 0
        elif(top_tuple_method == 'Top3'):
            top_tuple_nr = 3
            all_points_all_tuples_scores_df.loc[top_tuple_nr:, [one_score_column]] = 0
        elif(top_tuple_method == 'Top10'):
            top_tuple_nr = 10
            all_points_all_tuples_scores_df.loc[top_tuple_nr:, [one_score_column]] = 0
        elif(top_tuple_method == 'TopXPercent'):
            top_tuple_nr = int(all_points_all_tuples_scores_df.shape[0] * top_tuple_percentage)
            all_points_all_tuples_scores_df.loc[top_tuple_nr:, [one_score_column]] = 0

        one_weight_column = one_score_column.replace('score_', 'weight_')
        all_points_all_tuples_scores_df[one_weight_column] = all_points_all_tuples_scores_df[one_score_column] / all_points_all_tuples_scores_df[one_score_column].sum()
    if(using_jupyter_gui and debugging):
        print(all_points_all_tuples_scores_df.join(config_indexed_df.set_index('config_index'), on='config_index'))
    #all_points_all_tuples_scores_df = all_points_all_tuples_scores_df[weight_result_column_listing]
    all_points_all_tuples_scores_df.sort_values('config_index', inplace=True)
    return(all_points_all_tuples_scores_df)


# In[29]:


###t145325_expResult_train_selected_df.head()


# In[30]:


def test_one_point(one_test_index):
    seek_nr = one_test_index * combination_nr_per_time_point * item_size_bytes
    p_file = open(file_input_name, "rb")
    p_file.seek(seek_nr, os.SEEK_SET)
    t = np.fromfile(p_file, dtype=np.float32, count=combination_nr_per_time_point)
    t = pd.DataFrame(t.reshape((combination_nr_per_time_point_no_prediction, len(predict_step_length_listing) * data_variable_nr))) # 320 * 8
    t.columns = measurement_column_listing
    t['config_index'] = range(0, combination_nr_per_time_point_no_prediction) # 0, 320
    
    #t = t145325_expResult_df[np.isin(global_index_multiply_config_nr_listing, one_test_index)] # to-improve: use exact index directly
    #t = t.groupby(['config_index']).mean().reset_index().sort_values('config_index')
    #t145325_expResult_test_selected_df = t145325_expResult_test_selected_df.sort_values('config_index') # only one index, why not working ???
    
    if(measurement_metric != 'MAE'):
        stop('ERR: unexpected measurement_metric')
        quit()
    result_dict = {}
    for one_measurement_column in measurement_column_listing:
        one_weight_column = one_measurement_column.replace('err_', 'weight_')
        one_weighted_result_series = all_points_all_tuples_scores_weights_df[one_weight_column] * t[one_measurement_column]
        one_weighted_result = one_weighted_result_series.sum()
        one_result_column = one_measurement_column.replace('err_', 'weighted_err_')
        result_dict[one_result_column] = one_weighted_result
    result_df = pd.DataFrame(result_dict, index=[one_test_index,]) # must to have a index, can be anything: ref: https://stackoverflow.com/a/25326589
    return(result_df)


# In[31]:


def get_test_all_points_all_tuples_weighted_err(global_index_test_listing):
    if(debugging):
        if(not use_spark):
            all_points_all_tuples_result_listing = Parallel(n_jobs=parallel_job_nr)(delayed(test_one_point)(one_test_index) for one_test_index in global_index_test_listing[:10])
        else:
            my_rdd = sc.parallelize(list(global_index_test_listing[:3]))
            map_result = my_rdd.map(test_one_point)
            all_points_all_tuples_result_listing = map_result.collect()
    else:
        if(not use_spark):
            all_points_all_tuples_result_listing = Parallel(n_jobs=parallel_job_nr)(delayed(test_one_point)(one_test_index) for one_test_index in global_index_test_listing)
        else:
            my_rdd = sc.parallelize(list(global_index_test_listing), partition_nr)
            map_result = my_rdd.map(test_one_point)
            all_points_all_tuples_result_listing = map_result.collect()
    # the result: all_points_all_tuples_result_listing is a list of values
    all_points_all_tuples_weighted_err_df = pd.concat(all_points_all_tuples_result_listing)#.reset_index(drop=True) ???
    return(all_points_all_tuples_weighted_err_df)


# In[32]:


def get_metrics_from_weighted_err_df(all_points_all_tuples_weighted_err_df, label_state):
    t_avg = pd.DataFrame(all_points_all_tuples_weighted_err_df.mean()).T
    t_avg['metric'] = 'AVG'
    t_std = pd.DataFrame(all_points_all_tuples_weighted_err_df.std()).T
    t_std['metric'] = 'STD'
    t_mae = pd.DataFrame(all_points_all_tuples_weighted_err_df.abs().mean()).T
    t_mae['metric'] = 'MAE'
    t = t_avg.append(t_std).append(t_mae)
    t['cv_fold_order'] = cv_fold_order
    t['cv_fold_nr'] = cv_fold_nr
    t['label'] = label_state
    return(t)


# ## main loop

# In[48]:


config_indexed_df = pd.read_pickle('config_indexed_df.pkl')

from sklearn.model_selection import StratifiedKFold

stratified_k_fold = StratifiedKFold(n_splits=cv_fold_nr, shuffle=True, random_state=cv_random_seed)
stratified_k_fold_listing = stratified_k_fold.split(global_index_and_label_truncated_df['global_index'], global_index_and_label_truncated_df['label'])

if(save_predictions):
    weighted_err_of_all_folds_df = pd.DataFrame()
cv_fold_order = 0
results_of_all_folds_df = pd.DataFrame()
#train_index_of_truncated_global_index, test_index_of_truncated_global_index = next(stratified_k_fold_listing)
for train_index_of_truncated_global_index, test_index_of_truncated_global_index in stratified_k_fold_listing:


	# In[49]:


	cv_fold_order += 1
	print('======== cv_fold_order: ' + str(cv_fold_order) + ' of ' + str(cv_fold_nr) + ' ========')
	print('Timestamp new cv_fold_order: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	print("TRAIN-all:", train_index_of_truncated_global_index, "TEST-all:", test_index_of_truncated_global_index)
	global_index_train_listing = global_index_and_label_truncated_df['global_index'][train_index_of_truncated_global_index]
	global_index_test_listing  = global_index_and_label_truncated_df['global_index'][test_index_of_truncated_global_index]
	label_train_listing = global_index_and_label_truncated_df['label'][train_index_of_truncated_global_index]
	label_test_listing  = global_index_and_label_truncated_df['label'][test_index_of_truncated_global_index]

	# separate nonLabelled vs. Labelled, train, test
	global_index_train_non_labelled_listing = global_index_train_listing[label_train_listing == 0]
	global_index_train_labelled_listing = global_index_train_listing[label_train_listing == 1]
	global_index_test_non_labelled_listing = global_index_test_listing[label_test_listing == 0]
	global_index_test_labelled_listing = global_index_test_listing[label_test_listing == 1]

	# remove missing
	global_index_train_non_labelled_listing = np.setdiff1d(global_index_train_non_labelled_listing, missing_value_global_index_listing[0])
	global_index_train_labelled_listing = np.setdiff1d(global_index_train_labelled_listing, missing_value_global_index_listing[0])
	global_index_test_non_labelled_listing = np.setdiff1d(global_index_test_non_labelled_listing, missing_value_global_index_listing[0])
	global_index_test_labelled_listing = np.setdiff1d(global_index_test_labelled_listing, missing_value_global_index_listing[0])

	print('len(global_index_train_non_labelled_listing): ' + str(len(global_index_train_non_labelled_listing)))
	print('len(global_index_test_non_labelled_listing): ' + str(len(global_index_test_non_labelled_listing)))
	print('len(global_index_train_labelled_listing): ' + str(len(global_index_train_labelled_listing)))
	print('len(global_index_test_labelled_listing): ' + str(len(global_index_test_labelled_listing)))

	#if(not 't145325_expResult_df' in vars()):
	#    t145325_expResult_df = pd.read_pickle('/dev/shm/t145325_expResult_df.gsw.missingOnly406.pkl')


	# ## labelled_aware

	# In[50]:


	# train labelled_aware
	print('Timestamp train_labelled: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	scores_cache_file = cache_file_prefix + cache_config_string + '_fold_' + str(cv_fold_order) + '_labelled_scores_df.pkl'
	if(use_cache and os.path.isfile(scores_cache_file)):
		all_points_all_tuples_scores_df = pd.read_pickle(scores_cache_file)
	else:
		all_points_all_tuples_scores_df = get_train_all_points_all_tuples_scores_df(global_index_train_labelled_listing)
		if(use_cache):
			all_points_all_tuples_scores_df.to_pickle(scores_cache_file)
	train_labelled_scores_df = all_points_all_tuples_scores_df.copy()

	all_points_all_tuples_scores_weights_df = get_scores_weights_from_scores(all_points_all_tuples_scores_df)
	if(using_jupyter_gui):
		all_points_all_tuples_scores_weights_df
	if(save_verbose_results and not using_jupyter_gui):
		all_points_all_tuples_scores_weights_df.to_pickle('log/' + filename_prefix + '_fold_' + str(cv_fold_order) + '_' + str(cv_fold_nr) + '_labelled_scores_weights.pkl')


	# In[60]:


	# test labelled_aware
	print('Timestamp test_labelled: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	all_points_all_tuples_weighted_err_df = get_test_all_points_all_tuples_weighted_err(global_index_test_labelled_listing)
	if(save_predictions):
		t_err = all_points_all_tuples_weighted_err_df.copy()
		t_err['test_index'] = global_index_test_labelled_listing
		t_err['label_state'] = 'labelled_aware'
		weighted_err_of_all_folds_df = weighted_err_of_all_folds_df.append(t_err)

	labelled_all_points_all_tuples_weighted_err_df = all_points_all_tuples_weighted_err_df.copy() # for aware-combined

	t_metrics_df = get_metrics_from_weighted_err_df(all_points_all_tuples_weighted_err_df, label_state='labelled_aware')
	results_of_all_folds_df = results_of_all_folds_df.append(t_metrics_df)
	if(print_verbose_info):
		print(t_metrics_df)


	# ## non_labelled_aware 

	# In[38]:


	# train non_labelled_aware
	print('Timestamp train_non_labelled: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	scores_cache_file = cache_file_prefix + cache_config_string + '_fold_' + str(cv_fold_order) + '_non_labelled_scores_df.pkl'
	if(use_cache and os.path.isfile(scores_cache_file)):
		all_points_all_tuples_scores_df = pd.read_pickle(scores_cache_file)
	else:
		all_points_all_tuples_scores_df = get_train_all_points_all_tuples_scores_df(global_index_train_non_labelled_listing)
		if(use_cache):
			all_points_all_tuples_scores_df.to_pickle(scores_cache_file)
	train_non_labelled_scores_df = all_points_all_tuples_scores_df.copy()

	all_points_all_tuples_scores_weights_df = get_scores_weights_from_scores(all_points_all_tuples_scores_df)
	if(using_jupyter_gui):
		all_points_all_tuples_scores_weights_df
	if(save_verbose_results and not using_jupyter_gui):
		all_points_all_tuples_scores_weights_df.to_pickle('log/' + filename_prefix + '_fold_' + str(cv_fold_order) + '_' + str(cv_fold_nr) + '_non_labelled_scores_weights.pkl')


	# In[39]:


	# test non_labelled_aware
	print('Timestamp test_non_labelled: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	all_points_all_tuples_weighted_err_df = get_test_all_points_all_tuples_weighted_err(global_index_test_non_labelled_listing)
	if(save_predictions):
		t_err = all_points_all_tuples_weighted_err_df.copy()
		t_err['test_index'] = global_index_test_non_labelled_listing
		t_err['label_state'] = 'non_labelled_aware'
		weighted_err_of_all_folds_df = weighted_err_of_all_folds_df.append(t_err)

	t_metrics_df = get_metrics_from_weighted_err_df(all_points_all_tuples_weighted_err_df, label_state='non_labelled_aware')
	results_of_all_folds_df = results_of_all_folds_df.append(t_metrics_df)
	if(print_verbose_info):
		print(t_metrics_df)


	# ## aware (combined)

	# In[40]:


	print('Timestamp train & test aware: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	t_metrics_df = get_metrics_from_weighted_err_df(
		all_points_all_tuples_weighted_err_df.append(labelled_all_points_all_tuples_weighted_err_df), 
		label_state='both_aware')
	results_of_all_folds_df = results_of_all_folds_df.append(t_metrics_df)
	if(print_verbose_info):
		print(t_metrics_df)


	# ## unaware

	# In[270]:


	# train unaware
	print('Timestamp train unaware: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	all_points_all_tuples_scores_df = train_non_labelled_scores_df.append(train_labelled_scores_df).groupby('config_index').sum().reset_index().sort_values('config_index')
	all_points_all_tuples_scores_weights_df = get_scores_weights_from_scores(all_points_all_tuples_scores_df)
	all_points_all_tuples_scores_weights_df
	if(save_verbose_results and not using_jupyter_gui):
		all_points_all_tuples_scores_weights_df.to_pickle('log/' + filename_prefix + '_fold_' + str(cv_fold_order) + '_' + str(cv_fold_nr) + '_unaware_scores_weights.pkl')

	# test unaware
	## test labelled_unaware
	print('Timestamp labelled_unaware: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	all_points_all_tuples_weighted_err_df = get_test_all_points_all_tuples_weighted_err(global_index_test_labelled_listing)
	if(save_predictions):
		t_err = all_points_all_tuples_weighted_err_df.copy()
		t_err['test_index'] = global_index_test_labelled_listing
		t_err['label_state'] = 'labelled_unaware'
		weighted_err_of_all_folds_df = weighted_err_of_all_folds_df.append(t_err)
	t_metrics_df = get_metrics_from_weighted_err_df(all_points_all_tuples_weighted_err_df, label_state='labelled_unaware')
	results_of_all_folds_df = results_of_all_folds_df.append(t_metrics_df)
	if(print_verbose_info):
		print(t_metrics_df)
	## test non_labelled_unaware
	print('Timestamp non_labelled_unaware: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	all_points_all_tuples_weighted_err_df = get_test_all_points_all_tuples_weighted_err(global_index_test_non_labelled_listing)
	if(save_predictions):
		t_err = all_points_all_tuples_weighted_err_df.copy()
		t_err['test_index'] = global_index_test_non_labelled_listing
		t_err['label_state'] = 'non_labelled_unaware'
		weighted_err_of_all_folds_df = weighted_err_of_all_folds_df.append(t_err)
	t_metrics_df = get_metrics_from_weighted_err_df(all_points_all_tuples_weighted_err_df, label_state='non_labelled_unaware')
	results_of_all_folds_df = results_of_all_folds_df.append(t_metrics_df)
	if(print_verbose_info):
		print(t_metrics_df)
	## test mixed: non_labelled_unaware & labelled_aware
	print('Timestamp non_labelled_unaware: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	t_metrics_df = get_metrics_from_weighted_err_df(
		all_points_all_tuples_weighted_err_df.append(labelled_all_points_all_tuples_weighted_err_df), 
		label_state='mixed')
	results_of_all_folds_df = results_of_all_folds_df.append(t_metrics_df)
	if(print_verbose_info):
		print(t_metrics_df)
	## test both_unaware
	print('Timestamp both_unaware: ' + datetime.now().strftime("%Y%m%d_%H%M%S"))
	all_points_all_tuples_weighted_err_df = get_test_all_points_all_tuples_weighted_err(np.append(global_index_test_labelled_listing, global_index_test_non_labelled_listing))
	t_metrics_df = get_metrics_from_weighted_err_df(all_points_all_tuples_weighted_err_df, label_state='both_unaware')
	results_of_all_folds_df = results_of_all_folds_df.append(t_metrics_df)
	if(print_verbose_info):
		print(t_metrics_df)


	# In[285]:


	avg_results_of_all_folds_df = results_of_all_folds_df.loc[:, results_of_all_folds_df.columns.difference(['cv_fold_order'])].groupby(['label', 'metric']).mean().reset_index()
	print('until now, avg_results_of_all_folds_df: ')
	avg_results_of_all_folds_df.to_csv(sys.stdout)


	# In[273]:


	# main loop ended


# In[274]:


avg_results_of_all_folds_df = results_of_all_folds_df.loc[:, results_of_all_folds_df.columns.difference(['cv_fold_order'])].groupby(['label', 'metric']).mean().reset_index()
avg_results_of_all_folds_df['anomaly_method'] = anomaly_method
avg_results_of_all_folds_df['score_method'] = score_method
avg_results_of_all_folds_df['top_tuple_method'] = top_tuple_method

if(not using_jupyter_gui):
    avg_results_of_all_folds_df.to_pickle('log/' + filename_prefix + '_avg_results_of_all_folds_df.pkl')
    avg_results_of_all_folds_df.to_csv('log/' + filename_prefix + '_avg_results_of_all_folds_df.csv')
print('========= avg_results_of_all_folds_df: =========')
#avg_results_of_all_folds_df.to_csv(sys.stdout)

results_of_all_folds_df['anomaly_method'] = anomaly_method
results_of_all_folds_df['score_method'] = score_method
results_of_all_folds_df['top_tuple_method'] = top_tuple_method
if(not using_jupyter_gui):
    results_of_all_folds_df.to_pickle('log/' + filename_prefix + '_results_of_all_folds_df.pkl')
if(using_jupyter_gui or print_verbose_info):
    print('========= results_of_all_folds_df: =========')
    print(results_of_all_folds_df)

if(save_predictions):
    weighted_err_of_all_folds_df['param_string'] = '_'.join(sys.argv[1:])
    weighted_err_of_all_folds_df.to_pickle(mid_size_file_storage_path + filename_prefix + '_weighted_err_of_all_folds_df.pkl')

print('========= Exp ended: ' + datetime.now().strftime("%Y%m%d_%H%M") + ' =========')


# In[275]:


# TODO: add switch to compare: manual label vs. current Decomposition vs. SHESD etc.
# TODO: current status should be measured by near-history (t-1, ...), not self.
# TODO: add switch to compare: different measurement methods (mapping labels to status).
# TO-improve: file seek partial: ref is.gd/bRqZQy is.gd/KG5F66
# TO-improve: use other information, such as time of day, day of week, etc.
# discussion: there are other non-param detection methods // can put also SHESD, as long as the prediction is more accurate.
# future work: consider anomaly during neighbour searching.
# future work: distance.


# logics: 
#     overall MAE & weighted overall MAE (it is the best if anormaly-aware method can outperform in both metrics);
#     
