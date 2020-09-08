import numpy as np
import copy

class Dataset:
    def __init__(self, X, T, y,
                 variable_names, variable_weights,
                 experiment_start, experiment_end,
                 subject_IDs=None, subject_data=None,
                 variable_annotations={}):
        """ Store experimental data in an object.

        We assume that most preprocessing (variable selection, creation
        of aggregate data for higher taxa, rescaling and renormalization,
        etc.) has been done already, and that some prior on the
        likelihood of rules in the model applying to each variable
        has already been calculated.

        Arguments:

        X - list, containing for each subject an n_variables by
        n_timepoints_for_this_subject array of observations.
        T - list, containing for each subject a vector giving the
        observation timepoints (note the length will be different for
        each subject in general.)
        y - list/array of boolean or 0/1 values indicating whether each
        subject developed the condition of interest.
        variable_names - list of strings, used for formatting output
        variable_weights - array of weights for each variable which will
        be used later to calculate the prior probability that a rule applies
        to that variable.
        experiment_start - Time at which the experiment started.
        experiment_end - Time at which the experiment ended.
        subject_IDs - list of identifiers for each experimental subject
        (currently purely for reference)
        subject_data - Optional pandas dataframe giving additional information
        about each subject.

        variable_annotations: dict mapping variable names to taxonomic or
        other descriptions, optional

        This method sets up the following useful attributes:
        self.n_subjects
        self.n_variables
        Dataset objects also offer convenience methods
        apply_primitive, apply_rules, and stratify, which determine
        the output of primitives, rules, or rule lists applied to the
        data. This allows for various caching approaches that speed up
        these evaluations: currently implemented as an attribute
        _primitive_result_cache, a dict (empty by default) of arrays
        of booleans giving the truths of primitives, expressed as
        tuples, for each subject in the dataset.

        Raises ValueError if the number of variables reported
        on in each observation table in X differs from the number
        of variable names provided, or if that number does not
        match the dimension of the argument variable_weights.

        """
        self.X = X
        self.T = T
        self.y = np.array(y, dtype='bool')
        self.variable_names = variable_names
        self.variable_weights = variable_weights
        self.experiment_start = experiment_start
        self.experiment_end = experiment_end
        self.subject_data = subject_data
        self.subject_IDs = subject_IDs
        self.n_subjects = len(X)
        self.n_variables = len(variable_weights)
        self._primitive_result_cache = {}
        for array in X:
            # Allow a special case where a subject has no
            # observations
            if len(array) == 0:
                continue
            this_subject_n_variables, _ = array.shape
            if this_subject_n_variables != self.n_variables:
                raise ValueError('Observation-prior dimension mismatch.')
        if len(self.variable_names) != self.n_variables:
            raise ValueError('Incorrect number of variable names.')

        self.variable_annotations = variable_annotations

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        template = ('<Dataset with %d subjects ' +
                    '(%d observations of %d variables)>')
        n_obs = sum([len(timepoints) for timepoints in self.T])
        return template % (self.n_subjects, n_obs,
                           self.n_variables)


