""" Feature selection methods.
"""
import itertools
import numpy as np
from data_preprocessing.transforms import select_subjects, select_variables


def discard_where_data_missing(data, field):
    """ Discard subjects where data for a particular field is missing.

    Assumes the missing data value is NaN. Non-numeric values
    are never considered missing, even the empty string.

    """
    keep_indices = []
    for i, value in enumerate(data.subject_data[field].values):
        if not (np.isreal(value) and np.isnan(value)):
            keep_indices.append(i)
    return select_subjects(data, keep_indices)


def discard_low_overall_abundance(data, min_abundance_threshold,
                                  skip_variables=set()):
    """ Drop taxa whose summed data over all observations is too low.

    Specifically, for each variable (except those given by name in
    skip_variables), add together the data for each observation
    in each subjects. Discard the variable if this is less than
    min_abundance_threshold.

    Returns: a new Dataset object and an array of indices of variables
    _kept_ in the filtering process (to allow the same transformation to
    be performed on other data.)

    A ValueError will be raised if the selected conditions filter out
    all the variables.

    """
    keep_indices = []
    summed_data = np.zeros(data.n_variables)
    for subject_data in data.X:
        # recall timepoints are the columns of entries of X
        for timepoint in subject_data.T:
            summed_data += timepoint

    passing_variables = summed_data >= min_abundance_threshold
    for i,passes in enumerate(passing_variables):
        if passes or (data.variable_names[i] in skip_variables):
            keep_indices.append(i)

    return select_variables(data, keep_indices), keep_indices

def discard_low_abundance(data,
                          min_abundance_threshold,
                          min_consecutive_samples=2,
                          min_n_subjects=1,
                          skip_variables=set()):

    """ Drop taxa too rarely above threshold over consecutive time points.

    Specifically, for each variable (except those given by name in
    skip_variables), we count the number of subjects for which that
    variable exceeds min_abundance_threshold in at least
    min_consecutive_samples successive time points. If that number
    of subjects is less than min_n_subjects, the variable is
    dropped.

    Returns: a new Dataset object and an array of indices of variables
    _kept_ in the filtering process (to allow the same transformation to
    be performed on other data.)

    A ValueError will be raised if the selected conditions filter out
    all the variables.

    """
    keep_indices = []
    for i in range(data.n_variables):
        if data.variable_names[i] in skip_variables:
            keep_indices.append(i)
            continue
        n_passing_subjects = 0
        for subject_data in data.X:
            above_threshold = subject_data[i] > min_abundance_threshold
            if not any(above_threshold):
                continue
            # is the longest run longer than min_consecutive_samples?
            # This approach is overkill, but foolproof (and, once
            # you know what groupby does, readable:)
            run_lengths = [sum(group) for key,group in
                           itertools.groupby(above_threshold) if
                           key]
            if max(run_lengths) >= min_consecutive_samples:
                n_passing_subjects += 1
        if n_passing_subjects >= min_n_subjects:
            keep_indices.append(i)
    return select_variables(data, keep_indices), keep_indices


def discard_low_depth_samples(data, min_abundance_threshold):
    """ Drop observations where data summed over OTUs is too low.

    For example, we apply this when some samples had too low
    sequencing depth (or at least too few sequences which survived
    the pipeline.)

    For each subject, the data is added together along the OTU axis,
    and only those timepoints where the result is greater
    than threshold are kept.

    Returns: a new Dataset object.

    """
    new_data = data.copy()
    for i in range(new_data.n_subjects):
        table = new_data.X[i]
        if len(table) == 0:
            # Subjects with no observations are okay.
            continue
        depths = np.sum(table, axis=0)
        keep_indices = (depths >= min_abundance_threshold)
        table = table.T[keep_indices].T
        new_data.X[i] = table
        new_data.T[i] = new_data.T[i][keep_indices]
    return new_data

def trim(data, t0, t1):
    """ Drop observations before t0 or after t1.

    Returns a new Dataset object, with experiment_start set to t0
    and experiment_end set to t1.

    """
    new_data = data.copy()
    for i in range(new_data.n_subjects):
        times = new_data.T[i]
        table = new_data.X[i]
        keep_indices = (times >= t0) & (times <= t1)
        table = table.T[keep_indices].T
        new_data.X[i] = table
        new_data.T[i] = new_data.T[i][keep_indices]
    new_data.experiment_start = t0
    new_data.experiment_end = t1
    return new_data


def test_timepoints_0(timepoints, n_samples, n_intervals,
                      start, end, n_consecutive=1):
    # Likely a more efficient way to do this exists...
    boundaries = np.linspace(start, end, n_intervals+1)
    for i,t0 in enumerate(boundaries[:-n_consecutive]):
        t1=boundaries[i+n_consecutive]
        n_timepoints_in_window = np.sum(
            (t0 <= timepoints) &
            (timepoints <= t1)
        )
        if n_timepoints_in_window < n_samples:
            return False
    return True

def filter_on_sample_density(data, n_samples, interval,
                             method=0, n_consecutive=1):
    """Discard subjects with insufficient temporal sampling.

    In general, any subject without n_samples observations
    in certain time periods of length controlled by the
    'interval' argument will be dropped.

    If method=0, currently the only choice, divide the window
    [data.experiment_start, data.experiment_stop] into 'interval'
    equal pieces. If, in any continuous block of
    n_consecutive such pieces, there are fewer than
    n_samples observations, drop the subject.

    Returns a new dataset. n_subjects, subject_IDs, and subject_data
    will be updated appropriately (assuming subject_data is a
    pandas DataFrame: if not, it is left alone.)

    """
    keep_indices = []
    for i, timepoints in enumerate(data.T):
        if method == 0:
            test = test_timepoints_0
        else:
            raise ValueError
        okay = test(timepoints, n_samples, interval,
                    data.experiment_start,
                    data.experiment_end, n_consecutive=n_consecutive)
        if okay:
            keep_indices.append(i)
    return select_subjects(data, keep_indices)

def discard_surplus_internal_nodes(data):
    """ Drop inner nodes of variable tree not needed to maintain topology.

    All variables which are leaves of the tree, not nodes of the tree
    at all, or are needed to preserve relationships between the leaves,
    are kept.

    """
    keep_indices = []
    # This list won't include the root, but the root is going to
    # be kept by the pruning process anyway, so it doesn't matter.
    old_tree_nodes = {node.name for node in
                      data.variable_tree.get_descendants()}
    leaves = data.variable_tree.get_leaf_names()
    new_tree = data.variable_tree.copy()
    new_tree.prune(list(leaves), preserve_branch_length=True)
    new_tree_nodes = {node.name for node in new_tree.get_descendants()}

    for i, name in enumerate(data.variable_names):
        if name not in old_tree_nodes:
            keep_indices.append(i)
        elif name in new_tree_nodes:
            keep_indices.append(i)

    # Kludge solution for some slightly odd behavior
    # of the tree pruning routine, which keeps more nodes than are in
    # new_tree when prune(new_tree_nodes) is called

    new_data = select_variables(data, keep_indices)
    new_data.variable_tree = new_tree
    return new_data, keep_indices

def Average_transformX(data, N_i, N_c):
    """ Average and reformat data.
    Returns: an array containing the averaged values, of shape
    n_subjects x (n_variables * n_windows).
    """
    interval_edges = np.linspace(data.experiment_start, data.experiment_end, N_i+1)
    windows = list(zip(interval_edges, interval_edges[N_c:]))
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
    # for i in range(len(new_X)):
    #     feature_X.append(np.array(new_X[i]))
    new_X = np.reshape(new_X, (data.n_subjects, len(windows), data.n_variables))
    # print(new_X.shape)
    return new_X