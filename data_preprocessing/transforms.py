
import numpy as np
import pandas as pd
from data_preprocessing.data_object import Dataset


def select_subjects(dataset, keep_subject_indices, invert=False):
    """ Copy the dataset, retaining only specified 
    subjects. 

    Raises ValueError if keep_subject_indices is empty.
    If invert is True, keep all subjects _except_ those specified.
    dataset - rules.Dataset instance
    keep_subject_indices - list or array of numbers, the indices (into
    dataset.X/dataset.T/dataset.subject_IDs) of the subjects to be
    retained.
    """
    if len(keep_subject_indices) < 1:
        raise ValueError('No subjects to be kept.')

    if invert:
        exclude_indices = set(keep_subject_indices)
        keep_subject_indices = [i for i in range(dataset.n_subjects) if
                                i not in exclude_indices]
    new_data = dataset.copy()

    new_X = []
    new_T = []
    new_y = []
    new_subject_IDs = []
    for i in keep_subject_indices:
        new_X.append(new_data.X[i])
        new_T.append(new_data.T[i])
        new_y.append(new_data.y[i])
        new_subject_IDs.append(new_data.subject_IDs[i])
    new_data.X = new_X
    new_data.T = new_T
    new_data.y = np.array(new_y)
    new_data.subject_IDs = new_subject_IDs
    new_data.n_subjects = len(new_subject_IDs)
    if isinstance(new_data.subject_data, pd.DataFrame):
        new_data.subject_data = new_data.subject_data.loc[new_subject_IDs]
    new_data._primitive_result_cache = {}
    return new_data

def log_transform(data,zero_data_offset=1e-6,zero_tolerance=1e-10):
    new_data = data.copy()
    # We expect the data to be positive, so don't take the absolute
    # value before checking to see what is close to zero
    for i in range(len(new_data.X)):
        new_data.X[i][new_data.X[i]<zero_tolerance] = zero_data_offset
        new_data.X[i] = np.log(new_data.X[i])
    return new_data

def select_variables(dataset, keep_variable_indices):
    """ Copy the dataset, retaining only specified
    variables.

    Raises ValueError if keep_variable_indices is empty.

    Note that, if dataset has a variable_tree attribute,
    the tree will be pruned to keep only those nodes which are
    kept variables, and the additional nodes required to preserve the
    topology of the tree connecting them; thus, not all nodes in the
    resulting variable_tree are guaranteed to be variables.

    """
    if not keep_variable_indices:
       raise ValueError('No variables to be kept.')

    new_variable_names = [dataset.variable_names[i] for
                          i in keep_variable_indices]
    # We want to index into copies of the arrays in dataset.X
    # so that the underlying data is copied instead of referenced.
    temp_dataset = dataset.copy()
    new_X = []
    for subject_X in temp_dataset.X:
        if len(subject_X) == 0:
            new_X.append(subject_X)
        else:
            new_X.append(subject_X[keep_variable_indices])
    new_variable_weights = (
        temp_dataset.variable_weights[keep_variable_indices]
    )
    new_dataset = Dataset(
        new_X, temp_dataset.T, temp_dataset.y,
        new_variable_names, new_variable_weights,
        temp_dataset.experiment_start,
        temp_dataset.experiment_end,
        subject_IDs=temp_dataset.subject_IDs,
        subject_data=temp_dataset.subject_data,
        variable_annotations = temp_dataset.variable_annotations.copy()
    )
    if hasattr(temp_dataset, 'variable_tree'):
        new_tree = temp_dataset.variable_tree.copy()
        old_node_names = {n.name for n in new_tree.get_descendants()}
        new_nodes = [v for v in new_variable_names if v in old_node_names]
#        print 'debug select variables: new nodes:'
#        print new_nodes
        if new_nodes:
            new_tree.prune(new_nodes, preserve_branch_length=True)
            new_dataset.variable_tree = new_tree
        # Otherwise there is no point in retaining the tree as we
        # have dropped all variables with a tree relationship.
    return new_dataset

def take_relative_abundance(data):
    """ Transform abundance measurements to relative abundance. """

    new_data = data.copy()
    n_subjects = len(new_data.X)
    for i in range(n_subjects):
        abundances = new_data.X[i]
        # Data may be integer (counts): cast carefully
        # to float
        total_abundances = np.sum(abundances, axis=0).astype(np.float)
        relative_abundances = abundances/total_abundances
        new_data.X[i] = relative_abundances
    return new_data

def do_internal_normalization(data,
                              target_variable_names,
                              reject_threshold=1e-6):
    """ Normalize abundance measurements by sum of some variables. """
    try:
        target_indices = [data.variable_names.index(n) for n in
                            target_variable_names]
    except ValueError:
        raise ValueError(
            'Variable name %s specified for use in internal normalization,'
            ' but not found in data. Double-check it is a valid name, and'
            ' has not been accidentally removed by filtering settings.' )
    new_data = data.copy()
    n_subjects = len(new_data.X)
    for i in range(n_subjects):
        abundances = new_data.X[i]
        target_abundances = abundances[target_indices, :]
        # Data may be integer (counts): cast carefully
        # to float
        norm_factors = np.sum(target_abundances, axis=0).astype(np.float)
        if not np.all(norm_factors > reject_threshold):
            bad_indices = np.where(norm_factors <= reject_threshold)
            bad_timepoints = data.T[i][bad_indices]
            subject_id = data.subject_IDs[i]
            message = (
                'Error normalizing data for subject %s: '
                'sum of variables used for normalization is less than '
                'the minimum %.3g at timepoints %s'
                % (subject_id, reject_threshold,
                   ','.join(['%.3g' % t for t in bad_timepoints])
                )
            )
            raise ValueError(message)
        normalized_abundances = abundances/norm_factors
        new_data.X[i] = normalized_abundances
    return new_data


