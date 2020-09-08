import configparser
from sklearn.metrics import roc_auc_score, confusion_matrix
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegressionCV
import warnings
import argparse as ap
from scipy import stats
from sklearn.model_selection import KFold
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import catboost as cb
from numpy import *
from sklearn import *
from scipy.spatial.distance import hamming
import multiprocessing as mp
from sklearn.ensemble import GradientBoostingClassifier
from data_preprocessing import filtering, load_data, transforms
from tree import pplacer, taxonomy_annotation
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
import os 
import xgboost as xgb
from numpy import *
from sklearn import *
import pandas as pd
import plotly.graph_objs as go
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import cv2
import numpy as np
import operator
from pandas.core.common import flatten
import sys

def temporal_filter_if_needed(config, target_data):
    if config.has_option('preprocessing', 'temporal_abundance_threshold'):

        target_data, _ = filtering.discard_low_abundance(
            target_data,
            min_abundance_threshold=config.getfloat(
                'preprocessing',
                'temporal_abundance_threshold'),
            min_consecutive_samples=config.getfloat(
                'preprocessing',
                'temporal_abundance_consecutive_samples'),
            min_n_subjects=config.getfloat(
                'preprocessing',
                'temporal_abundance_n_subjects')
        )
    return target_data

def log_transform_if_needed(config, target_data):
    if config.has_option('preprocessing', 'log_transform'):
        if config.getboolean('preprocessing','log_transform'):
            target_data = log_transform(target_data)
    return target_data

def log_transform(data,zero_data_offset=1e-6,zero_tolerance=1e-10):
    new_data = data.copy()
    # We expect the data to be positive, so don't take the absolute
    # value before checking to see what is close to zero
    for i in range(len(new_data.X)):
        new_data.X[i][new_data.X[i]<zero_tolerance] = zero_data_offset
        new_data.X[i] = np.log(new_data.X[i])
    return new_data

def get_normalization_variables(config, data):
    """ Figure out which variables to normalize by, if relevant.
    
    If preprocessing/normalization_variables_file is set, loads that
    file and reads a variable name from each line.

    Returns a list of variable names (as strings).

    """
    if (config.has_option('preprocessing', 'normalization_variables_file')
        and
        config.has_option('preprocessing', 'normalize_by_taxon')):
        raise ValueError('Mutually exclusive normalization options given.')
    
    if config.has_option('preprocessing', 'normalization_variables_file'):
        filename = config.get(
            'preprocessing',
            'normalization_variables_file'
        )
        with open(filename) as f:
            variables = [s.strip() for s in f.readlines()]
        # Tolerate extra newlines, etc
        result = [v for v in variables if v]
    
        return result

    elif config.has_option('preprocessing','normalize_by_taxon'):
        if not config.has_option('data','placement_table'):
            raise ValueError(
                'A taxonomic placement table must be '
                'specified to allow normalization by a particular taxon.'
            )
        placement_table_filename = config.get('data','placement_table')
        if config.has_option('data','sequence_key'):
            sequence_fasta_filename = config.get('data','sequence_key')
        else:
            sequence_fasta_filename = None
        table = taxonomy_annotation.load_table(
            placement_table_filename,
            sequence_fasta_filename
        )
        target_taxon = config.get('preprocessing','normalize_by_taxon')
        target_variables = []
        for v in data.variable_names:
            classifications = table.loc[v,:]
            if target_taxon in classifications.values:
                target_variables.append(v)
        if not target_variables:
            raise ValueError(
                'Found no variables in designated normalization '
                'taxon "%s".' % target_taxon
            )
        prefix = config.get('description', 'tag')
        fname = prefix + '_variables_used_for_normalization.csv'
        subtable = table.loc[target_variables,:]
        subtable.to_csv(fname)
        
        return target_variables
    else:
        raise ValueError(
            'Must set normalization_variables_file or '
            'normalization taxon in section '
            '"preprocessing" to do internal normalization.'
        )

def preprocess_step1(config):
    """ Load data, apply initial filters and convert to RA, create Dataset object.
    """
    # 1. Input files.
    counts_file = config.get('data', 'abundance_data')
    metadata_file = config.get('data', 'sample_metadata')
    subject_file = config.get('data', 'subject_data')
    if config.has_option('data', 'sequence_key'):
        sequence_file = config.get('data', 'sequence_key')
    else:
        sequence_file = None

    # 2. Outcome
    outcome_variable = config.get('data', 'outcome_variable')
    outcome_positive_value = config.get('data', 'outcome_positive_value')

    try:
        outcome_positive_value = int(outcome_positive_value)
    except ValueError:
        if outcome_positive_value.lower() in ('true', 'false'):
            message = ('Boolean outcome values should specified as 1 or 0 '
                       '(not the strings "true", "false", or derivatives of these) '
                       '- the currently specified value will be interpreted as a '
                       'generic categorical value which will probably lead to '
                       'undesirable behavior!')
            warnings.warn(message)
        pass

    # Loading the data.
    data_type = '16s'
    if config.has_option('data', 'data_type'):
        data_type = config.get('data', 'data_type').lower()
    assert data_type in ('16s', 'metaphlan')
    if data_type == 'metaphlan':
        do_weights = False
        weight_scale = 1.0
        if config.has_option('data', 'metaphlan_do_weights'):
            do_weights = config.getboolean('data', 'metaphlan_do_weights')
        if config.has_option('data', 'metaphlan_weight_scale'):
            weight_scale = config.getfloat('data', 'metaphlan_weight_scale')
        data = load_data.load_metaphlan_result(
            counts_file,
            metadata_file,
            subject_file,
            do_weights=do_weights,
            weight_scale=weight_scale,
            outcome_variable=outcome_variable,
            outcome_positive_value=outcome_positive_value
        )

    else:  # default to assuming 16s
        data = load_data.load_16S_result(
            counts_file,
            metadata_file,
            subject_file,
            sequence_id_filename=sequence_file,
            outcome_variable=outcome_variable,
            outcome_positive_value=outcome_positive_value
        )

    # 3. Filtering
    # 3a. Overall abundance filter
    if config.has_option('preprocessing', 'min_overall_abundance'):
        # Drop sequences/OTUs with fewer reads (summing across all
        # samples) than the threshold
        minimum_reads_per_sequence = config.getfloat(
            'preprocessing', 'min_overall_abundance'
        )
        data, _ = filtering.discard_low_overall_abundance(
            data,
            minimum_reads_per_sequence
        )

    # # 3b. Sample depth filter
    if config.has_option('preprocessing', 'min_sample_reads'):
        # Drop all samples where the total number of reads was below a
        # threshold
        minimum_reads_per_sample = config.getfloat(
            'preprocessing',
            'min_sample_reads'
        )
        data = filtering.discard_low_depth_samples(
            data,
            minimum_reads_per_sample
        )

    # # 3c. Trimming the experimental window
    if config.has_option('preprocessing', 'trim_start'):
        experiment_start = config.getfloat('preprocessing', 'trim_start')
        experiment_end = config.getfloat('preprocessing', 'trim_stop')
        data = filtering.trim(data, experiment_start, experiment_end)


    # 3d. Drop subjects with inadequately dense temporal sampling
    if config.has_option('preprocessing', 'density_filter_n_samples'):
        subject_min_observations_per_long_window = (
            config.getfloat('preprocessing',
                            'density_filter_n_samples')
        )
        n_intervals = config.getint(
            'preprocessing',
            'density_filter_n_intervals')
        n_consecutive = config.getint(
            'preprocessing',
            'density_filter_n_consecutive'
        )
        data = filtering.filter_on_sample_density(
            data,
            subject_min_observations_per_long_window,
            n_intervals,
            n_consecutive=n_consecutive
        )

    # 3e. Relative abundance transformation, or other normalization.


    if config.has_option('preprocessing', 'take_relative_abundance'):
        if config.getboolean('preprocessing', 'take_relative_abundance'):
            data = transforms.take_relative_abundance(data)

    if config.has_option('preprocessing', 'do_internal_normalization'):
        if config.getboolean('preprocessing','do_internal_normalization'):
            normalization_variables = get_normalization_variables(config, data)
            if config.has_option('preprocessing',
                                'internal_normalization_min_factor'):
                threshold = config.getfloat(
                    'preprocessing',
                    'internal_normalization_min_factor'
                )
            else:
                threshold = 1.0
            data = transforms.do_internal_normalization(
                data, normalization_variables, reject_threshold=threshold
            ) 

    return data

def preprocess_step2(config, data):
    """ Aggregate, optionally transform, temporal abundance filter, etc.
    Taxonomy information is applied here (after the tree
    is established)
    """
    # 3f. Phylogenetic aggregation.
    #
    # print("3start==================", data.X[0].shape)
    has_tree = False
    if config.has_option('preprocessing', 'aggregate_on_phylogeny'):
        if config.getboolean('preprocessing', 'aggregate_on_phylogeny'):

            jplace_file = config.get('data', 'jplace_file')
            data, _, _ = pplacer.aggregate_by_pplacer_simplified(
                jplace_file,
                data
            )
    has_tree = True


    if has_tree and config.has_option('data','taxonomy_source'):
        # Valid options are 'pplacer' and 'table' and 'hybrid'
        taxonomy_source = config.get('data','taxonomy_source')
       
    else:
        taxonomy_source = None

    if taxonomy_source == 'table':
        placement_table_filename = config.get('data','placement_table')
        if config.has_option('data','sequence_key'):
            sequence_fasta_filename = config.get('data','sequence_key')
        else:
            sequence_fasta_filename = None
        taxonomy_annotation.annotate_dataset_table(
            data,
            placement_table_filename,
            sequence_fasta_filename
        )

    elif taxonomy_source == 'pplacer':
        jplace_filename = config.get('data', 'jplace_file')
        taxa_table_filename = config.get('data', 'pplacer_taxa_table')
        seq_info_filename = config.get('data', 'pplacer_seq_info')
        taxonomy_annotation.annotate_dataset_pplacer(
            data,
            jplace_filename,
            taxa_table_filename,
            seq_info_filename
        )

    elif taxonomy_source == 'hybrid':
        jplace_filename = config.get('data', 'jplace_file')
        taxa_table_filename = config.get('data', 'pplacer_taxa_table')
        seq_info_filename = config.get('data', 'pplacer_seq_info')
        placement_table_filename = config.get('data','placement_table')
        if config.has_option('data','sequence_key'):
            sequence_fasta_filename = config.get('data','sequence_key')
        else:
            sequence_fasta_filename = None
        taxonomy_annotation.annotate_dataset_hybrid(
            data,
            jplace_filename,
            taxa_table_filename,
            seq_info_filename,
            placement_table_filename,
            sequence_fasta_filename
        )
        
    # 3g. Log transform
    data = log_transform_if_needed(config, data)

    # 3h. Temporal abundance filter.
    # We drop all variables except those which exceed a threshold
    # abundance for a certain number of consecutive observations in a
    # certain number of subjects
    data = temporal_filter_if_needed(config, data)

    # 3i. Surplus internal node removal.
    if config.has_option('preprocessing', 'discard_surplus_internal_nodes'):
        if config.getboolean('preprocessing',
                             'discard_surplus_internal_nodes'):
          
            data, _ = filtering.discard_surplus_internal_nodes(data)
          

    return data

def mean_transform(data, N_i, N_c, otus_len):
   
    new_y = 1 * data.y
    variable_ordering = data.variable_names[:]

    interval_edges = np.linspace(data.experiment_start, data.experiment_end, N_i + 1)
    windows = list(zip(interval_edges, interval_edges[N_c:]))
    
    print('Time Windows: ', windows)
    print('Time start: ', data.experiment_start)
    print('Time end: ', data.experiment_end)

    
    
    new_X = np.zeros((data.n_subjects, len(windows) * data.n_variables))
    for subject_index in range(data.n_subjects):
        column_index = 0
        subject_data = data.X[subject_index]
        subject_timepoints = data.T[subject_index]
        for t0, t1 in windows:
            relevant_time_indices = (subject_timepoints >= t0) & (subject_timepoints <= t1)
            relevant_data = subject_data[:, relevant_time_indices]
            if len(relevant_data) == 0:
                raise ValueError('Cannot classify subject %d: no points within time window %.3f-%.3f.' %
                                 (subject_index, t0, t1))
            for variable_index in range(data.n_variables):
                average = np.mean(relevant_data[variable_index])
                new_X[subject_index, column_index] = average
                column_index += 1
                
    Flatten_X = new_X
                
    new_X = np.reshape(new_X, (data.n_subjects, data.n_variables, len(windows)))
    
    Otu_X =new_X[:,:otus_len,:]
    
    Phy_X =new_X[:,otus_len:,:]
    
    
    
    
    return Otu_X, Phy_X, new_X, Flatten_X, new_y , variable_ordering

def run_from_config_file(filename):
    config = configparser.ConfigParser()
   
    config.read(filename)

    if config.has_section('benchmarking'):
      
    
        ra_data = preprocess_step1(config)
    

        comparison_data_1 = log_transform_if_needed(config, ra_data)  
        

        comparison_data_2 = temporal_filter_if_needed(config,log_transform_if_needed(config, ra_data))


        comparison_data_3 = preprocess_step2(config, ra_data)
        
        

        comparison_n_intervals = config.getint('comparison_methods',
                                           'n_intervals')
        comparison_n_consecutive = config.getint('comparison_methods',
                                             'n_consecutive')
        
        OtuX, PhyX, newX, FlattenX, y , var = mean_transform(comparison_data_3, comparison_n_intervals, 
            comparison_n_consecutive, len(comparison_data_2.variable_names))
        
        
        
        return  OtuX, PhyX, newX, FlattenX, y , var
 
def save_npy(file_name,data):
    np.save(file=file_name+".npy", arr=data)

def read_params(args):
    parser = ap.ArgumentParser(description='Experiment')
    arg = parser.add_argument
    arg('-fn', '--fn', type=str, help='datasets')
    return vars(parser.parse_args())



if __name__ == "__main__":
    par = read_params(sys.argv)
    fn= str(par['fn'])

    file_name="cfg_file/" + fn + "/" + fn+ "_benchmark.cfg"

    OtuX, PhyX, newX, FlattenX, label, var =run_from_config_file(file_name)
    print('Comb: ',newX.shape)
    print('Label length: ', len(label))

    save_npy(fn+'_Combine_Matrix', newX)
    save_npy(fn+'_Label', label)

