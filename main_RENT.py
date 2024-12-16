# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import seaborn as sns
import sys
import time
import warnings
import hoggorm as ho
import hoggormplot as hopl

from abc import ABC, abstractmethod
from itertools import combinations, combinations_with_replacement
from joblib import Parallel, delayed

from sklearn.linear_model import LogisticRegression, ElasticNet, \
    LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score, \
                            matthews_corrcoef, r2_score, accuracy_score, \
                            log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxnetSurvivalAnalysis

from scipy.stats import t


class RENT_Base(ABC):
    """
    The constructor initializes common variables of RENT_Classification and RENT_Regression.
    Initializations that are specific for classification or regression are described in 
    detail in RENT for binary classification and RENT for regression, respectively.
    
    PARAMETERS
    -----
    data: <numpy array> or <pandas dataframe>
        Dataset on which feature selection shall be performed. 
        Variable types must be numeric or integer.
    target: <numpy array> or <pandas dataframe>
        Response variable of data.
    feat_names : <list>
        List holding feature names. Preferably a list of string values. 
        If empty, feature names will be generated automatically. 
        Default: ``feat_names=[]``.
    C : <list of int or float values>
        List with regularisation parameters for ``K`` models. The lower,
        the stronger the regularization is. Default: ``C=[1,10]``.
    l1_ratios : <list of int or float values>
        List holding ratios between l1 and l2 penalty. Values must be in [0,1]. For
        pure l2 use 0, for pure l1 use 1. Default: ``l1_ratios=[0.6]``.
    autoEnetParSel : <boolean>
        Cross-validated elastic net hyperparameter selection.
            - ``autoEnetParSel=True`` : peform a cross-validation pre-hyperparameter\
                search, such that RENT runs only with one hyperparamter setting.
            - ``autoEnetParSel=False`` : perform RENT with each combination of ``C`` \
                and ``l1_ratios``. Default: ``autoEnetParSel=True``.
    BIC : <boolean>
        Use the Bayesian information criterion to select hyperparameters.
            - ``BIC=True`` : use BIC to select RENT hyperparameters.
            - ``BIC=False``: no use of BIC.   
    poly : <str> 
        Create non-linear features. Default: ``poly='OFF'``.
            - ``poly='OFF'`` : no feature interaction.
            - ``poly='ON'`` : feature interaction and squared features (2-polynoms).
            - ``poly='ON_only_interactions'`` : only feature interactions, \
                no squared features.
    testsize_range : <tuple float>
         Inside RENT, ``K`` models are trained, where the testsize defines the 
         proportion of train data used for testing of a single model. The testsize 
         can either be randomly selected inside the range of ``testsize_range`` 
         for each model or fixed by setting the two tuple entries to the same value. 
         The tuple must be in range (0,1). Default: ``testsize_range=(0.2, 0.6)``.
    K : <int>
        Number of unique train-test splits. Default ``K=100``.
    scale : <boolean>
        Columnwise standardization each of the K train datasets. Default ``scale=True``.
    random_state : <None or int>
        Set a random state to reproduce your results. Default: ``random_state=None``.
            - ``random_state=None`` : no random seed. 
            - ``random_state={0,1,2,...}`` : random seed set.       
    verbose : <int>
        Track the train process if value > 1. If ``verbose = 1``, only the overview
        of RENT input will be shown. Default: ``verbose=0``.
    """
    __slots__=["_data", "_target", "_feat_names", "_C", "_l1_ratios", "_autoEnetParSel",
               "_BIC", "_poly", "_testsize_range", "_K", "_scale", "_random_state",
               "_verbose", "_summary_df", "_score_dict", "_BIC_df", "_best_C",
               "_best_l1_ratio", "_indices", "_polynom", "_runtime", "_scores_df", "_combination", 
               "_zeros", "_perc", "_self_var", "_X_test", "_test_data", "_zeros_df","_sel_var",
               "_incorrect_labels", "_pp_data"]

    def __init__(self, data, target, feat_names=[], C=[1,10], l1_ratios = [0.6],
                 autoEnetParSel=True, BIC=False, poly='OFF',testsize_range=(0.2, 0.6), 
                 K=100, scale = True, random_state = None, verbose = 0):

        if any(c < 0 for c in C):
            sys.exit('C values must not be negative!')
        if any(l < 0 for l in l1_ratios) or any(l > 1 for l in l1_ratios):
            sys.exit('l1 ratios must be in [0,1]!')
        if autoEnetParSel not in [True, False]:
            sys.exit('autoEnetParSel must be True or False!')
        if BIC not in [True, False]:
            sys.exit('BIC must be True or False!')
        if scale not in [True, False]:
            sys.exit('scale must be True or False!')
        if poly not in ['ON', 'ON_only_interactions', 'OFF']:
            sys.exit('Invalid poly parameter!')
        if K<=0:
            sys.exit('Invalid K!')
        if K<10:
            # does not show warning...
            warnings.warn('Attention: K is very small!', DeprecationWarning)
        if len(target.shape) == 2 :
            target = target.values

        # Print parameters if verbose = True
        if verbose == 1:
            print('data dimension:', np.shape(data), ' data type:', type(data))
            print('target dimension:', np.shape(target))
            print('regularization parameters C:', C)
            print('elastic net l1_ratios:', l1_ratios)
            print('poly:', poly)
            print('number of models in ensemble:', K)
            print('random state:', random_state)
            print('verbose:', verbose)


        # Define all objects needed later in methods below
        self._target = target
        self._K = K
        self._feat_names = feat_names
        self._testsize_range = testsize_range
        self._scale = scale
        self._verbose = verbose
        self._autoEnetParSel = autoEnetParSel
        self._BIC = BIC
        self._random_state = random_state
        self._poly = poly

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, list):
                data.index = list(data.index)
            self._indices = data.index
        else:
            self._indices = list(range(data.shape[0]))

        if isinstance(self._target, pd.Series):
            self._target.index = self._indices

        # If no feature names are given, create some
        if len(self._feat_names) == 0:
            print('No feature names found - automatic generate feature names.')

            for ind in range(1, np.shape(data)[1] + 1):
                self._feat_names.append('f' + str(ind))

        # Extend data if poly was set to 'ON' or 'ON_only_interactions'
        if self._poly == 'ON':
            self._polynom = PolynomialFeatures(interaction_only=False, \
                                         include_bias=False)
            self._data = self._polynom.fit_transform(data)
            polynom_comb = list(combinations_with_replacement(self._feat_names,\
                                                              2))
            polynom_feat_names = []
            # Construct a new name for squares and interactions
            for item in polynom_comb:
                name = item[0] + '^2' if item[0] == item[1] else item[0] + '*' + item[1]
                polynom_feat_names.append(name)

            flist = list(self._feat_names)
            flist.extend(polynom_feat_names)
            self._feat_names = flist
            self._data = pd.DataFrame(self._data)
            self._data.index = self._indices
            self._data.columns = self._feat_names

        elif self._poly == 'ON_only_interactions':
            self._polynom = PolynomialFeatures(interaction_only=True,\
                                         include_bias=False)
            self._data = self._polynom.fit_transform(data)

            polynom_comb = list(combinations(self._feat_names, 2))
            polynom_feat_names = []

            # Construct a new name for squares and interactions
            for item in polynom_comb:
                name = item[0] + '*' + item[1]
                polynom_feat_names.append(name)

            flist = list(self._feat_names)
            flist.extend(polynom_feat_names)
            self._feat_names = flist
            self._data = pd.DataFrame(self._data)
            self._data.index = self._indices
            self._data.columns = self._feat_names

        elif self._poly == 'OFF':
            self._data = pd.DataFrame(data)
            self._data.index=self._indices
            self._data.columns = self._feat_names

        else:
            sys.exit('Value for paramter "poly" not regcognised.')


        if self._autoEnetParSel == True:
            if self._BIC == False:
                self._C, self._l1_ratios = self._par_selection(C=C, 
                                                            l1_ratios=l1_ratios)
            else:
                self._C, self._l1_ratios = self._par_selection_BIC(C=C, 
                                                            l1_ratios=l1_ratios)
            self._C = [self._C]
            self._l1_ratios = [self._l1_ratios]
        else:
            self._C = C
            self._l1_ratios = l1_ratios
    
    @abstractmethod
    def run_parallel(self, K):
        pass

    @abstractmethod
    def _par_selection(self, C_params, l1_params, n_splits, testsize_range):
        pass
    
    @abstractmethod
    def _par_selection_BIC(self, C_params, l1_params, n_splits, testsize_range):
        pass

    @abstractmethod
    def get_summary_objects(self):
        pass
    
    @abstractmethod
    def _prepare_validation_study(self, test_data, test_labels, num_drawings, 
                                  num_permutations, metric='mcc', alpha=0.05):
        pass

    def train(self):
        """
        If ``autoEnetParSel=False``, this method trains ``K`` * ``len(C)`` 
        * ``len(l1_ratios)`` models in total. 
        The number of models using the same hyperparamters is ``K``.
        Otherwise, if the best parameter combination is selected with 
        cross-validation, only ``K`` models are trained.
        For each model elastic net regularisation is applied for feature selection. 
        Internally, ``train()`` calls the ``run_parallel()`` function for classification 
        or regression, respectively.
        """
        np.random.seed(0)
        self._random_testsizes = np.random.uniform(self._testsize_range[0],
                                                  self._testsize_range[1],
                                                  self._K)

        # Initiate dictionaries. Keys are (C, K, num_w_init)
        self._weight_dict = {}
        self._score_dict = {}
        self._weight_list = []
        self._score_list = []

        # stop runtime
        start = time.time()
        # Call parallelization function
        Parallel(n_jobs=-1, verbose=0, backend='threading')(
             map(delayed(self.run_parallel), range(self._K)))
        ende = time.time()
        self._runtime = ende-start

        # find best parameter setting and matrices
        result_list=[]
        for l1 in self._l1_ratios:
            for C in self._C:
                spec_result_list = []
                for k in self._score_dict.keys():
                    if k[0] == C and k[1] ==l1:
                        spec_result_list.append(self._score_dict[k])
                result_list.append(spec_result_list)

        means=[]
        for r in range(len(result_list)):
            means.append(np.mean(result_list[r]))

        self._scores_df = pd.DataFrame(np.array(means).reshape(\
                                  len(self._l1_ratios), \
                                  len(self._C)), \
        index= self._l1_ratios, columns = self._C)

        self._zeros_df = pd.DataFrame(index = self._l1_ratios,\
                                   columns=self._C)
        for l1 in self._l1_ratios:
            for C in self._C:
                count = 0
                for K in range(self._K):
                    nz = \
                    len(np.where(pd.DataFrame(self._weight_dict[(C, l1, K)\
])==0)[0])
                    count = count + nz / len(self._feat_names)
                count = count / (self._K)
                self._zeros_df.loc[l1, C] = count

        if len(self._C)>1 or len(self._l1_ratios)>1:
            normed_scores = pd.DataFrame(self._min_max(
                self._scores_df.copy().values))
            normed_zeros = pd.DataFrame(self._min_max(
                self._zeros_df.copy().values))
            normed_zeros = normed_zeros.astype('float')
            self._combination = 2 * ((normed_scores.copy().applymap(self._inv) + \
                                        normed_zeros.copy().applymap(
                                            self._inv)).applymap(self._inv))
        else:
            self._combination = 2 * ((self._scores_df.copy().applymap(self._inv) + \
                                 self._zeros_df.copy().applymap(
                                     self._inv)).applymap(self._inv))
        self._combination.index = self._scores_df.index.copy()
        self._combination.columns = self._scores_df.columns.copy()

        self._scores_df.columns.name = 'Scores'
        self._zeros_df.columns.name = 'Zeros'
        self._combination.columns.name = 'Harmonic Mean'

        best_row, best_col  = np.where(
            self._combination == np.nanmax(self._combination.values))
        self._best_l1_ratio = self._combination.index[np.nanmax(best_row)]
        self._best_C = self._combination.columns[np.nanmin(best_col)]

    def select_features(self, tau_1_cutoff=0.9, tau_2_cutoff=0.9, tau_3_cutoff=0.975):
        """
        Selects features based on the cutoff values for tau_1_cutoff, 
        tau_2_cutoff and tau_3_cutoff.
        
        Parameters
        ----------
        tau_1_cutoff : <float>
            Cutoff value for tau_1 criterion. Choose value between 0 and
            1. Default: ``tau_1=0.9``.
        tau_2_cutoff : <float>
            Cutoff value for tau_2 criterion. Choose value between 0 and
            1. Default:``tau_2=0.9``.
        tau_3_cutoff : <float>
            Cutoff value for tau_3 criterion. Choose value between 0 and
            1. Default: ``tau_3=0.975``.
            
        Returns
        -------
        <numpy array>
            Array with selected features.
        """
        if not hasattr(self, '_best_C'):
            sys.exit('Run train() first!')

        #Loop through all K models
        weight_list = [self._weight_dict[(self._best_C,
                                                 self._best_l1_ratio,
                                                 K)] for K in range(self._K)]
        weight_array = np.vstack(weight_list)

        #Compute results based on weights
        counts = np.count_nonzero(weight_array, axis=0)
        self._perc = counts / len(weight_list)
        means = np.mean(weight_array, axis=0)
        stds = np.std(weight_array, axis=0)
        signum = np.apply_along_axis(self._sign_vote, 0, weight_array)
        t_test = t.cdf(
            abs(means / np.sqrt((stds ** 2) / len(weight_list))), \
                (len(weight_list)-1))

        # Conduct a dataframe that stores the results for the criteria
        summary = np.vstack([self._perc, signum, t_test])
        self._summary_df = pd.DataFrame(summary)
        self._summary_df.index = ['tau_1', 'tau_2', 'tau_3']
        self._summary_df.columns = self._feat_names

        self._sel_var = np.where(
                (self._summary_df.iloc[0, :] >= tau_1_cutoff) &
                (self._summary_df.iloc[1, :] >= tau_2_cutoff) &
                (self._summary_df.iloc[2, :] >= tau_3_cutoff\
                            ))[0]

        #if len(self._sel_var) == 0:
        #    warnings.warn("Attention! Thresholds are too restrictive - no features selected!")
        return self._sel_var
    
    def BIC_cutoff_search(self, parameters):
        """
        Compute the Bayesian information criterion for each combination of tau1, tau2 and tau3.
        
        PARAMETERS
        -----
        parameters: <dict> or
            Cutoff parameters to evaluate.
        Returns
        -------
        <numpy array>
            Array wth the BIC values.
        """
        sc = StandardScaler()
        # Bayesian information criterion
        BIC = np.zeros(shape=(len(parameters['t1']), len(parameters['t2']),
                                len(parameters['t3'])))
        # grid search t1, t2, t3
        for i, t1 in enumerate(parameters['t1']):
            for j, t2 in enumerate(parameters['t2']):
                for k, t3 in enumerate(parameters['t3']):
                    sel_feat = self.select_features(t1, t2, t3)

                    train_data = sc.fit_transform(self._data.iloc[:,sel_feat])
                    lr = LogisticRegression().fit(train_data, self._target)
                    num_params = len(np.where(lr.coef_ != 0)[1]) + 1
                    pred_proba = lr.predict_proba(train_data)
                    pred = lr.predict(train_data)
                    
                    log_lik = log_loss(y_true=self._target, y_pred=pred_proba, normalize=False)
                    B = 2 * log_lik + np.log(len(pred)) * num_params
                    BIC[i,j,k] = B
                    
        return BIC

    def get_summary_criteria(self):
        """
        Summary statistic of the selection criteria tau_1, tau_2 and 
        tau_3 (described in ``select_features()``)
        for each feature. All three criteria are in [0,1] .
        
        RETURNS
        -------
        <pandas dataframe>
            Matrix where rows represent selection criteria and 
            columns represent features.
        """
        if not hasattr(self, '_summary_df'):
            sys.exit('Run select_features() first!')
        return self._summary_df

    def get_weight_distributions(self, binary = False):
        """
        In each of the ``K`` models, feature weights are fitted, i.e. 
        an individiual weight is assigned feature 1 for model 1, 
        model 2, up to model ``K``. This method returns the weight 
        for every feature and model (1:``K``) combination.
        
        PARAMETERS
        ----------
        binary : <boolean>
            Default: ``binary=False``.
                - ``binary=True`` : binary matrix where entry is 1 \
                    for each weight unequal to 0.
                - ``binary=False`` : original weight matrix.
                
        RETURNS
        -------
        <pandas dataframe>
            Weight matrix. Rows represent models (1:K), 
            columns represents features.
        """
        if not hasattr(self, '_weight_dict'):
            sys.exit('Run train() first!')

        # weights_df = pd.DataFrame()
        weights_df_list = []
        for k in self._weight_dict.keys():
            if k[0] == self._best_C and k[1] == self._best_l1_ratio:
                # weights_df = weights_df.append( \
                #         pd.DataFrame(self._weight_dict[k]))
                weights_df_list.append(pd.DataFrame(self._weight_dict[k])) # append to df deprecated in pandas 1.4

        weights_df = pd.concat(weights_df_list, ignore_index=True)
        weights_df.index = ['mod {0}'.format(x+1) for x in range(len(weights_df_list))]
        weights_df.columns = self._feat_names
        
        if binary == True:
            return((weights_df != 0).astype(np.int_))
        else:
            return(weights_df)

    def get_scores_list(self):
        """
        Prediction scores over the ``K`` models.
        RETURNS
        -------
        <list>
            Scores list.
        """
        return [
            self._score_dict[k]
            for k in self._score_dict.keys()
            if k[0] == self._best_C and k[1] == self._best_l1_ratio
        ]

    def get_enetParam_matrices(self):
        """
        Three pandas data frames showing result for all combinations
        of ``l1_ratio`` and ``C``.
        
        RETURNS
        -------
        <list> of <pandas dataframes>
            - dataFrame_1: holds average scores for \
                predictive performance.
            - dataFrame_2: holds average percentage of \
                how many feature weights were set to zero.
            - dataFrame_3: holds harmonic means between \
                dataFrame_1 and dataFrame_2.
        """
        if not hasattr(self, '_weight_dict'):
            sys.exit('Run train() first!')
        return self._scores_df, self._zeros_df, self._combination

    def get_cv_matrices(self):
        """
        Three pandas data frames showing cross-validated result for all combinations
        of ``C`` and ``l1_ratio`` . Only applicable if ``autoEnetParSel=True``.
        
        RETURNS
        -------
        <list> of <pandas dataframes>
            - dataFrame_1: average scores for predictive performance. \
                The higher the score, the better the parameter combination. 
            - dataFrame_2: average percentage of how many feature weights \
                are set to zero. The higher the average percentage, the stronger \
                    the feature selection with the corresponding paramter combination.
            - dataFrame_3: harmonic means between normalized dataFrame_1 and normalized \
                dataFrame_2. The parameter combination with the highest \
                    harmonic mean is selected.
        """
        if self._autoEnetParSel == True and self._BIC ==False:
            return self._scores_df_cv, self._zeros_df_cv, self._combination_cv
        else:
            print("autoEnetParSel=False or BIC=True - parameters have not been selected with cross-validation.")

    def get_BIC_matrix(self):
        """
        Dataframe with BIC value for each combination of ``C`` and ``11_ratio``.
        RETURNS
        -------
        <pandas dataframes>
            Dataframe of BIC values.
        """
        if self._autoEnetParSel == True and self._BIC ==True:
            return self._BIC_df
        else:
            print("BIC=False - parameters have not been selected with BIC.")
    
    def get_enet_params(self):
        """
        Get current hyperparameter combination of ``C`` and ``l1_ratio`` that 
        is used in RENT analyses. By default it is the best combination found. 
        If `autoEnetParSel=False` the user can change the combination 
        with ``set_enet_params()``. 
        
        RETURNS
        -------
        <tuple>
            A tuple (C, l1_ratio).
        """
        if not hasattr(self, '_best_C'):
            sys.exit('Run train() first!')
        return self._best_C, self._best_l1_ratio
    
    def get_runtime(self):
        """
        Total RENT training time in seconds.
        
        RETURNS
        -------
        <numeric value>
            Time.
        """
        return self._runtime

    def set_enet_params(self, C, l1_ratio):
        """
        Set hyperparameter combination of ``C`` and ``l1_ratio``, 
        that is used for analyses. Only useful if ``autoEnetParSel=False``.
        
        PARAMETERS
        ----------
        C: <float>
            Regularization parameter.
        l1_ratio: <float>
            l1 ratio with value in [0,1]. 
        """
        
        if (C not in self._C) | (l1_ratio not in self._l1_ratios):
            sys.exit('No weights calculated for this combination!')
        self._best_C = C
        self._best_l1_ratio = l1_ratio

    def plot_selection_frequency(self):
        """
        Barplot of tau_1 value for each feature.
        """
        if not hasattr(self, '_perc'):
            sys.exit('Run select_features() first!')

        plt.figure(figsize=(10, 7))
        (markers, stemlines, baseline) = plt.stem(self._perc
        # , use_line_collection=True %% This is always True in matplotlib 3.8.4+
        )
        plt.setp(markers, marker='o', markersize=5, color='black',
            markeredgecolor='darkorange', markeredgewidth=0)
        plt.setp(stemlines, color='darkorange', linewidth=0.5)
        plt.show()

    def plot_elementary_models(self):
        """
        Two lineplots where the first curve shows the prediction score over 
        ``K`` models. The second curve plots the percentage of weights set 
        to 0, respectively.
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        num_zeros = np.sum(1 - self.get_weight_distributions(binary=True), \
                            axis=1) / len(self._feat_names)
        
        scores = self.get_scores_list()
        data = pd.DataFrame({"num_zeros" : num_zeros, "scores" : scores})
        
        plt.plot(data.num_zeros.values, linestyle='--', marker='o', \
                 label="% zero weights")
        plt.plot(data.scores.values, linestyle='--', marker='o', label="score")
        plt.legend()
        ax.set_xlabel('elementary models')
        plt.title("Analysis of ensemble models")

    
    def plot_object_PCA(self, cl=0, comp1=1, comp2=2, 
                        problem='class', hoggorm=True, 
                        hoggorm_plots=[1,2,3,4,6], sel_vars=True):
        """
        PCA analysis. For classification problems, PCA can be computed either 
        on a single class separately or on both classes. Different coloring 
        possibilities for the scores are provided.
        Besides scores, loadings, correlation loadings, biplot, and explained 
        variance plots are available. 
        
        Parameters
        ----------
        cl : <int>, <str>
            Perform PCA on cl. Default: ``cl=0``.
                - ``cl=0``: Class 0.
                - ``cl=1``: Class 1.
                - ``cl='both'``: All objects (incorrect predictions coloring).
                - ``cl='continuous'``: All objects (gradient coloring). \
                    For classification problems, this is the only valid option.
        comp1 : <int>
            First PCA component to plot. Default: ``comp1=1``.            
        comp2 : <int>
            Second PCA component to plot. Default: ``comp2=2``.  
        problem : <str>
            Classification or regression problem. Default: ``problem='class'``.
                - ``problem='class'``: Classification problem. Can be used with \
                    all possible ``cl`` inputs.
                - ``problem='regression'``: Regression problem. \
                    Can only be used with ``cl='continuous'``.
        hoggorm : <boolean>
            To not use plots from hoggormplot package, set ``hoggorm=False``. \
                Default: ``hoggorm=True``.
        hoggorm_plots : <list>
            Choose which plots from hoggormplot are plotted. Only plots that are \
                relevant for RENT are possible options. ``hoggorm=True`` must be set. \
                    Default: ``hoggorm_plots=[1,2,3,4,6]``.
                - 1: scores plot
                - 2: loadings plot
                - 3: correlation loadings plot
                - 4: biplot
                - 6: explained variance plot
        sel_vars : <boolean>
            Only use the features selected with RENT for PCA. Default: ``sel_vars=True``.            
        """
        if cl not in [0, 1, 'both', 'continuous']:
            sys.exit(" 'cl' must be either 0, 1, 'both' or 'continuous'")
        if problem not in ['class', 'regression']:
            sys.exit(" 'problem' must be either 'class' or 'regression' ")
        if not hasattr(self, '_sel_var'):
            sys.exit('Run select_features() first!')
        if not hasattr(self, '_incorrect_labels'):
            sys.exit('Run get_summary_objects() first!')
        if problem == "regression" and cl != "continuous":
            sys.exit("The input is invalid. For 'problem = regression', 'cl' \
                     must be 'continuous' ")
        # catch if classification with continuous. (check if RENT class or RENT reg)

        if cl != 'continuous':
            dat = pd.merge(self._data, self._incorrect_labels.iloc[:,[1,-1]], \
                                 left_index=True, right_index=True)
            if sel_vars == True:
                variables = list(self._sel_var)
                variables.extend([-2,-1])
        else:

            if problem == "regression":
                dat = pd.merge(self._data, self._incorrect_labels.iloc[:,-1], \
                                         left_index=True, right_index=True)
            else:
                obj_mean = pd.DataFrame(np.nanmean( \
                        self.get_object_probabilities(), 1), \
                    index=self._data.index)
                obj_mean.columns = ["pred_means"]
                dat = pd.merge(self._data, obj_mean, \
                                         left_index=True, right_index=True)
            if sel_vars == True:
                variables = list(self._sel_var)
                variables.extend([-1])

        if sel_vars == True:
            if cl in ['both', 'continuous']:
                data = dat.iloc[:,variables]
            else:
                data = dat.iloc[np.where(dat.iloc[:,-2]==cl)[0],variables]
        else:
            data = dat
        if cl != 'continuous':
            data = data.sort_values(by='% incorrect')
            pca_model = ho.nipalsPCA(arrX=data.iloc[:,:-2].values, \
                                       Xstand=True, cvType=['loo'])
        else:
            pca_model = ho.nipalsPCA(arrX=data.iloc[:,:-1].values, \
                                       Xstand=True, cvType=['loo'])

        scores = pd.DataFrame(pca_model.X_scores())
        scores.index = list(data.index)
        scores.columns = ['PC{0}'.format(x+1) for x in \
                                 range(pca_model.X_scores().shape[1])]
        scores['coloring'] = data.iloc[:,-1]

        XexplVar = pca_model.X_calExplVar()
        var_comp1 = round(XexplVar[comp1-1], 1)
        var_comp2 = round(XexplVar[comp2-1], 1)


        fig, ax = plt.subplots()
        ax.set_xlabel('comp ' + str(comp1) +' ('+str(var_comp1)+'%)', fontsize=10)
        ax.set_ylabel('comp ' + str(comp2) +' ('+str(var_comp2)+'%)', fontsize=10)
        ax.set_title('Scores plot', fontsize=10)
        ax.set_facecolor('silver')

        # Find maximum and minimum scores along the two components
        xMax = max(scores.iloc[:, (comp1-1)])
        xMin = min(scores.iloc[:, (comp1-1)])

        yMax = max(scores.iloc[:, (comp2-1)])
        yMin = min(scores.iloc[:, (comp2-1)])

        # Set limits for lines representing the axes.
        # x-axis
        if abs(xMax) >= abs(xMin):
            extraX = xMax * .4
            limX = xMax * .3

        else:
            extraX = abs(xMin) * .4
            limX = abs(xMin) * .3

        if abs(yMax) >= abs(yMin):
            extraY = yMax * .4
            limY = yMax * .3

        else:
            extraY = abs(yMin) * .4
            limY = abs(yMin) * .3

        xMaxLine = xMax + extraX
        xMinLine = xMin - extraX
        yMaxLine = yMax + extraY
        yMinLine = yMin - extraY

        ax.plot([0, 0], [yMaxLine, yMinLine], color='0.4', linestyle='dashed',
                linewidth=3)
        ax.plot([xMinLine, xMaxLine], [0, 0], color='0.4', linestyle='dashed',
                linewidth=3)

        # Set limits for plot regions.
        xMaxLim = xMax + limX
        xMinLim = xMin - limX
        yMaxLim = yMax + limY
        yMinLim = yMin - limY
        ax.set_xlim(xMinLim, xMaxLim)
        ax.set_ylim(yMinLim, yMaxLim)

        # plot
        if cl == 0:
            plt.scatter(scores['PC'+str(comp1)], scores['PC'+str(comp2)],
                        c= scores['coloring'], cmap='Greens', marker ="^")
            cbar = plt.colorbar()
            cbar.set_label('% incorrect predicted class 0', fontsize=10)
        elif cl == 1:
            plt.scatter(scores['PC'+str(comp1)], scores['PC'+str(comp2)],
                        c= scores['coloring'], cmap='Reds')
            cbar = plt.colorbar()
            cbar.set_label('% incorrect predicted class 1', fontsize=10)
        elif cl == 'both':
            zeros = np.where(data.iloc[:,-2]==0)[0]
            ones = np.where(data.iloc[:,-2]==1)[0]

            plt.scatter(scores.iloc[zeros,(comp1-1)],
                        scores.iloc[zeros,(comp2-1)],
                        c= scores.iloc[zeros,-1], cmap='Greens', marker="^",
                        alpha=0.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('% incorrect predicted class 0', fontsize=10)
            plt.scatter(scores.iloc[ones,(comp1-1)],
                        scores.iloc[ones,(comp2-1)],
                        c= scores.iloc[ones,-1], cmap='Reds', alpha=0.5)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('% incorrect predicted class 1', fontsize=10)

            mlist = []
            col_list = []

            for i in range(len(data.index)):
                if data.iloc[i,-2]==0:
                    mlist.append("^")
                else:
                    mlist.append("o")

            for i in range(len(data.index)):
                if data.iloc[i,-2]==0 and data.iloc[i,-1]==0:
                    col_list.append('honeydew')
                elif data.iloc[i,-2]==1 and data.iloc[i,-1]==0:
                    col_list.append('snow')
                elif data.iloc[i,-2]==0 and data.iloc[i,-1]>0 and data.iloc[i,-1]<50:
                    col_list.append('mediumspringgreen')
                elif data.iloc[i,-2]==1 and data.iloc[i,-1]>0 and data.iloc[i,-1]<50:
                    col_list.append('tomato')
                elif data.iloc[i,-2]==0 and data.iloc[i,-1]>=50 and data.iloc[i,-1]<100:
                    col_list.append('green')
                elif data.iloc[i,-2]==1 and data.iloc[i,-1]>=50 and data.iloc[i,-1]<100:
                    col_list.append('red')
                elif data.iloc[i,-2]==0 and data.iloc[i,-1]==100:
                    col_list.append('darkgreen')
                elif data.iloc[i,-2]==1 and data.iloc[i,-1]==100:
                    col_list.append('maroon')
                else:
                    col_list.append(np.nan)

            for i in range(len(mlist)):
                plt.scatter(scores.iloc[i,(comp1-1)], scores.iloc[i,(comp2-1)],
                            marker=mlist[i], c=col_list[i])

        elif cl == 'continuous':
            plt.scatter(scores.iloc[:,(comp1-1)], scores.iloc[:,(comp2-1)],
                        c=scores.iloc[:,-1],
                        cmap='YlOrRd')
            cbar = plt.colorbar()
            if problem == "class":
                cbar.set_label('average object prediction', fontsize=10)
            else:
                cbar.set_label('mean absolute error', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        objnames = list(data.index.astype('str'))
        if hoggorm == True:
            if cl != 'continuous':
                hopl.plot(pca_model, plots=hoggorm_plots, comp = [comp1,comp2],
                        objNames=objnames, XvarNames=list(data.columns[:-2]))
            else:
                hopl.plot(pca_model, plots=hoggorm_plots, comp = [comp1,comp2],
                        objNames=objnames, XvarNames=list(data.columns[:-1]))

    def plot_validation_study(self, test_data, test_labels, num_drawings, 
                              num_permutations, metric='mcc', alpha=0.05):
        """
        Two validation studies based on a Student's `t`-test. \
            The null-hypotheses claim that
            - RENT is not better than random feature selection.
            - RENT performs equally well on the real and a randomly permutated target.
            
        If ``poly='ON'`` or ``poly='ON_only_interactions'`` in the RENT initialization, 
        the test data is automatically polynomially transformed.
        
        PARAMETERS
        ----------
        
        test_data : <numpy array> or <pandas dataframe>
            Dataset, used to evalute predictive models in the validation study.
            Must be independent of the data, RENT is computed on.
        test_lables: <numpy array> or <pandas dataframe>
            Response variable of test_data.
        num_drawings: <int>
            Number of independent feature subset drawings for VS1.
        num_permutations: <int>
            Number of independent test_labels permutations for VS2.
        metric: <str>
            The metric to evaluate ``K`` models. Default: ``metric='mcc'``. 
            Only relevant for classification tasks. For regression R2-score is used.
            
                - ``scoring='accuracy'`` :  Accuracy
                - ``scoring='f1'`` : F1-score
                - ``scoring='precision'`` : Precision
                - ``scoring='recall'``: Recall
                - ``scoring='mcc'`` : Matthews Correlation Coefficient
        alpha: <float>
            Significance level for the `t`-test. Default ``alpha=0.05``.
        """
        if not hasattr(self, '_sel_var'):
            sys.exit('Run select_features() first!')

        if self._poly != 'OFF':
            test_data = pd.DataFrame(self._polynom.fit_transform(test_data))
            test_data.columns = self._data.columns
            self._test_data = test_data
        
        score, VS1, VS2 = self._prepare_validation_study(test_data, 
                                                         test_labels, 
                                                         num_drawings, 
                                                         num_permutations,
                            metric='mcc', alpha=0.05)

        heuristic_p_value_VS1 = sum(VS1 > score) / len(VS1)
        T = (np.mean(VS1) - score) / (np.std(VS1,ddof=1) / np.sqrt(len(VS1)))
        print("mean VS1", np.mean(VS1))
        p_value_VS1 = t.cdf(T, len(VS1)-1)
        print("VS1: p-value for average score from random feature drawing: ", 
              p_value_VS1)
        print("VS1: heuristic p-value (how many scores are higher than" +
              " the RENT score): ", heuristic_p_value_VS1)

        if p_value_VS1 <= alpha:
            print('With a significancelevel of ', alpha, ' H0 is rejected.')
        else:
            print('With a significancelevel of ', alpha, ' H0 is accepted.')
        print(' ')
        print('-----------------------------------------------------------')
        print(' ')
   
        heuristic_p_value_VS2 = sum(VS2 > score) / len(VS2)
        print("Mean VS2", np.mean(VS2))
        T = (np.mean(VS2) - score) / (np.std(VS2,ddof=1) / np.sqrt(len(VS2)))
        p_value_VS2 = t.cdf(T, len(VS2)-1)
        print("VS2: p-value for average score from permutation of test labels: ", 
              p_value_VS2)
        print("VS2: heuristic p-value (how many scores are higher"+
              " than the RENT score): ", heuristic_p_value_VS2)
        if p_value_VS2 <= alpha:
            print('With a significancelevel of ', alpha, ' H0 is rejected.')
        else:
            print('With a significancelevel of ', alpha, ' H0 is accepted.')

        plt.figure(figsize=(15, 7))
        sns.kdeplot(VS1, shade=True, color="b", label='VS1')
        sns.kdeplot(VS2, shade=True, color="g", label='VS2')
        plt.axvline(x=score, color='r', linestyle='--',
                    label='RENT prediction score')
        plt.legend(prop={'size': 12})
        plt.ylabel('density', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title('Validation study', fontsize=18)

    def _inv(self, num):
        """
        Invert a numeric value unequal to 0.
        
        PARAMETERS
        ----------
        <float>
            ``num``: numeric value
            
        RETURNS
        -------
        <numeric value>
            Inverted value. 
        """
        if num == 0:
            return np.inf
        elif num == np.inf:
            return 0
        else:
            return num ** -1

    def _sign_vote(self, arr):
        """
        Calculate tau_2.
        
        PARAMETERS
        ----------
        <numpy array>
            ``arr``: Array of numeric values.
            
        RETURNS
        -------
        <numeric value>
            Inverted value. 
        """
        return np.abs(np.sum(np.sign(arr))) / len(arr)

    def _min_max(self, arr):
        """
        Min-max standardization. 
        
        PARAMETERS
        ----------
        <numpy array>
            ``arr``: Array of numeric values.
            
        RETURNS
        -------
        <numpy array>
            1-d array or matrix of higher dimension.
        """
        return (arr-np.nanmin(arr)) / (np.nanmax(arr)-np.nanmin(arr))

    def _poly_transform_testdata(self, test_data):
        """
        Generate polynomial features for a test dataset.
        
        PARAMETERS
        ----------
        <numpy array> or <pandas DataFrame>
            ``arr``: Array or DataFrame of numeric values.
            
        RETURNS
        -------
        <pandas DataFrame>
            Transformed test dataset.
        """
        return pd.DataFrame(self._polynom.fit_transform(test_data), columns = self._data.columns)
 
class CoxRENT(RENT_Base):
    """
    This class carries out RENT on a given survival analysis dataset using the Cox Proportional Hazards model.
    
    PARAMETERS
    ----------
    data : <numpy array> or <pandas dataframe>
        Dataset on which feature selection is performed. \
        Variable types must be numeric or integer.
    target : <structured numpy array> or <pandas dataframe>
        Response variable of data containing time and event status.
    feat_names : <list>
        List holding feature names. Preferably a list of string values. \
        If empty, feature names will be generated automatically. \
        Default: ``feat_names=[]``.
    C : <list of int or float values>
        List with regularization parameters for ``K`` models. The lower,
        the stronger the regularization is. Default: ``C=[1,10]``.
    l1_ratios : <list of int or float values>
        List holding ratios between l1 and l2 penalty. Values must be in [0,1]. \
        For pure l2 use 0, for pure l1 use 1. Default: ``l1_ratios=[0.6]``.
    autoEnetParSel : <boolean>
        Cross-validated elastic net hyperparameter selection.
        - ``autoEnetParSel=True`` : perform a cross-validation pre-hyperparameter \
          search, such that RENT runs only with one hyperparameter setting.
        - ``autoEnetParSel=False`` : perform RENT with each combination of ``C`` \
          and ``l1_ratios``. Default: ``autoEnetParSel=True``.
    poly : <str> 
        Create non-linear features. Default: ``poly='OFF'``.
        - ``poly='OFF'`` : no feature interaction.
        - ``poly='ON'`` : feature interaction and squared features (2-polynoms).
        - ``poly='ON_only_interactions'`` : only feature interactions, \
          no squared features.
    testsize_range : <tuple float>
        Inside RENT, ``K`` models are trained, where the testsize defines the \
        proportion of train data used for testing of a single model. The testsize 
        can either be randomly selected inside the range of ``testsize_range`` for \
        each model or fixed by setting the two tuple entries to the same value. 
        The tuple must be in range (0,1).
        Default: ``testsize_range=(0.2, 0.6)``.
    K : <int>
        Number of unique train-test splits. Default: ``K=100``.
    scale : <boolean>
        Columnwise standardization of the ``K`` train datasets. \
        Default: ``scale=True``.
    random_state : <None or int>
        Set a random state to reproduce your results. \
        Default: ``random_state=None``.
        - ``random_state=None`` : no random seed. 
        - ``random_state={0,1,2,...}`` : random seed set.
    verbose : <int>
        Track the train process if value > 1. If ``verbose = 1``, only the overview
        of RENT input will be shown. Default: ``verbose=0``.
        
    RETURNS
    ------
    <class>
        A class that contains the RENT survival analysis model based on Cox Proportional Hazards.
    """
    
    __slots__ = ["_data", "_target", "_feat_names", "_C", "_l1_ratios", "_autoEnetParSel",
                 "_BIC", "_poly", "_testsize_range", "_K", "_scale", "_random_state",
                 "_verbose", "_summary_df", "_score_dict", "_BIC_df", "_best_C",
                 "_best_l1_ratio", "_indices", "_polynom", "_random_testsizes", "_runtime", "_scores_df", "_combination", 
                 "_zeros", "_perc", "_self_var", "_X_test", "_test_data", "_sel_var", "_weight_dict", "_scores_df_cv", 
                 "_zeros_df_cv", "_combination_cv", "_weight_list", "_score_list", "_BIC_list", "_BIC_score_list", "_BIC_zeros_list",
                 "_predictions", "_test_indices", "_histogram_data", "_incorrect_labels"
                 ]

    def __init__(self, data, target, feat_names=[], C=[1,10], l1_ratios = [0.6],
                 autoEnetParSel=True, BIC=False, poly='OFF',
                 testsize_range=(0.2, 0.6), K=100, scale=True, random_state=None, 
                 verbose=0):

        # Initialize new attributes
        self._predictions = {}
        self._test_indices = {}
        self._histogram_data = {}
        self._incorrect_labels = {}

        # Give the target array the proper field names 'event' and 'time'
        old_names = target.dtype.names  # Extract original field names
        old_dtypes = [target.dtype[name] for name in old_names]  # Extract data types

        new_dtype = []
        for name, dtype in zip(old_names, old_dtypes):
            if np.issubdtype(dtype, np.bool_) or np.issubdtype(dtype, np.integer):  
                new_name = 'event'  # Boolean or integer type for 'event'
            elif np.issubdtype(dtype, np.number):  
                new_name = 'time'  # Numeric type (covers both float and int) for 'time'
            else:
                new_name = name  # Preserve the original name if dtype doesn't match expected types
            new_dtype.append((new_name, dtype))

        new_dtype = np.dtype(new_dtype)
        new_target = np.empty(target.shape, dtype=new_dtype)

        for old_name, new_name in zip(old_names, new_dtype.names):
            new_target[new_name] = target[old_name]

        target = new_target

        

        super().__init__(data, target, feat_names, C, l1_ratios, 
                         autoEnetParSel, BIC, poly, testsize_range, K, scale, 
                         random_state, verbose)
        
        print(f"Initialized CoxRENT with _testsize_range: {self._testsize_range} and K: {self._K}")
        # self.model = CoxnetSurvivalAnalysis()

    def _par_selection(self, 
                       C, 
                       l1_ratios, 
                       n_splits=5, 
                       testsize_range=(0.25,0.25)):
        """
        Preselect best 'C' and 'l1_ratio' with cross-validation for CoxnetSurvivalAnalysis.

        PARAMETERS
        ----------
        C : <list of int or float values>
            List with regularization parameters for 'K' models. The lower, 
            the stronger the regularization is.
        l1_ratios : <list of int or float values>
            List holding ratios between l1 and l2 penalty. Values must be in [0,1].
            For pure l2 use 0, for pure l1 use 1.
        n_splits : <int>
            Number of splits for cross-validation. Default: ``n_splits=5``.
        testsize_range : <tuple float>
            Range of test sizes for cross-validation. Default: ``testsize_range=(0.25,0.25)``.
            Testsize can be fixed by setting both entries to the same value.

        RETURNS
        -------
        <tuple>
            First entry: suggested `C` parameter.
            Second entry: suggested `l1 ratio`.

        """      
        if self._random_state is not None:
            kf = KFold(n_splits=n_splits, random_state=self._random_state, shuffle=True)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True)
        scores_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C)
        zeros_df = pd.DataFrame(np.zeros, index=l1_ratios, columns=C)

        def run_parallel(l1):
            """
            Parallel computation for 'K' * 'C' * 'l1_ratios' models.

            PARAMETERS
            ----------
            l1 : <float>
                current l1 ratio in the parallelization framework.
            """
            for reg in C:
                scores = []
                zeros = []
                for train, test in kf.split(self._data, self._target):
                    # Find the parameters that are 0
                    if self._scale == True:
                        sc = StandardScaler()
                        train_data = sc.fit_transform(self._data.iloc[train,:])
                        train_target = self._target[train]
                        test_data_split = sc.transform(self._data.iloc[test,:])
                        test_target = self._target[test]
                    elif self._scale == False:
                        train_data = self._data.iloc[train,:].values
                        train_target = self._target[train]
                        test_data_split = self._data.iloc[test,:].values
                        test_target = self._target[test]

                    # Fit the Cox model
                    model = CoxnetSurvivalAnalysis(alphas=[reg], l1_ratio=l1,
                                                   max_iter=5000,
                                                #    random_state=self._random_state,
                                                   normalize=False)
                    
                    model.fit(train_data, train_target)

                    # Select the non-zero coefficients
                    mod_coef = model.coef_.reshape(1, len(model.coef_))
                    params = np.where(mod_coef != 0)[1]

                    # if there are paramteres != 0, build a prediction model and
                    # find best parameter combination w.r.t. scoring
                    if len(params) == 0:
                        scores.append(np.nan)
                        zeros.append(np.nan)
                    else:
                        zeros.append((len(self._data.columns)-len(params))\
                                     /len(self._data.columns))

                        train_data_1 = train_data[:, params]
                        test_data_1 = test_data_split[:, params]

                        model.fit(train_data_1, train_target)

                        # Using concordance index for scoring
                        pred_risks = model.predict(test_data_1)
                        c_index = concordance_index_censored(test_target['event'], test_target['time'], pred_risks)[0]
                        scores.append(c_index)

                        zeros.append((len(self._data.columns)-len(params))/len(self._data.columns))

                scores_df.loc[l1, reg] = np.nanmean(scores)
                zeros_df.loc[l1, reg] = np.nanmean(zeros)

        Parallel(n_jobs=-1, verbose=1, backend="threading")(
            map(delayed(run_parallel), l1_ratios))
        
        s_arr = scores_df.stack()
        if len(np.unique(s_arr))==1:
            best_row, best_col = np.where(zeros_df.values == np.nanmax(zeros_df.values))
            best_l1 = zeros_df.index[np.nanmax(best_row)]
            best_C = zeros_df.columns[np.nanmax(best_col)]
        else:
            normed_scores = pd.DataFrame(self._min_max(scores_df.values))
            normed_zeros = pd.DataFrame(self._min_max(zeros_df.values))

            combination = 2 * ((normed_scores.copy().applymap(self._inv) + \
                                normed_zeros.copy().applymap(self._inv)
                                ).applymap(self._inv))
            combination.index = scores_df.index.copy()
            combination.columns = scores_df.columns.copy()
            best_combination_row, best_combination_col = np.where(combination == np.nanmax(combination.values))

            best_l1 = combination.index[np.nanmax(best_combination_row)]
            best_C = combination.columns[np.nanmax(best_combination_col)]

        self._scores_df_cv, self._zeros_df_cv, self._combination_cv = \
            scores_df, zeros_df, combination
        
        self._scores_df_cv.columns.name = 'Scores'
        self._zeros_df_cv.columns.name = 'Zeros'
        self._combination_cv.columns.name = 'Harmonic mean'
        return(best_C, best_l1)
    
    def _par_selection_BIC(self, C, l1_ratios):
        """
        Placeholder implementation for BIC-based parameter selection.
        """
        pass
    
    def _prepare_validation_study(self, test_data, test_labels, num_drawings, num_permutations, metric='mcc', alpha=0.05):
        """
        Placeholder implementation for preparing validation study.
        """
        # Remember! Swap out mcc for something else if you use this
        pass

    def run_parallel(self, K):
        """
        If ``autoEnetParSel=False``, parallel computation of ``K`` * ``len(C)`` * \
            ``len(l1_ratios)`` linear regression models. Otherwise, \
                computation of ``K`` models.
        
        PARAMETERS
        -----
        K: 
            Range of train-test splits. The parameter cannot be set directly \
                by the user but is used for an internal parallelization. 
        """
        for C in self._C:
            for l1 in self._l1_ratios:

                if self._random_state is None:
                    X_train, X_test, y_train, y_test = train_test_split(
                        self._data, self._target,
                        test_size=self._random_testsizes[K],
                        stratify=self._target['event'],)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                    self._data, self._target,
                    test_size=self._random_testsizes[K],
                    stratify=self._target['event'],
                    random_state=self._random_state)

                self._X_test = X_test

                if self._scale == True:
                    sc = StandardScaler()
                    X_train_std = sc.fit_transform(X_train)
                    X_test_std = sc.transform(X_test)
                else:
                    X_train_std = X_train.copy().values
                    X_test_std = X_test.copy().values

                if self._verbose > 1:
                    print('l1 = ', l1, 'C = ', C, ', TT split = ', K) 

                model = CoxnetSurvivalAnalysis(
                    alphas=[C], 
                    l1_ratio=l1,
                    max_iter=5000, 
                    # random_state=self._random_state,
                    normalize=False
                    ).fit(X_train_std, y_train)

                mod_coef = model.coef_.reshape(1, len(model.coef_))
                self._weight_dict[(C, l1, K)] = mod_coef
                self._weight_list.append(mod_coef)
                
                pred_risks = model.predict(X_test_std)

                # Calculate and store the concordance index as the score
                c_index = concordance_index_censored(y_test['event'], y_test['time'], pred_risks)[0]
                self._score_dict[(C, l1, K)] = c_index
                self._score_list.append(c_index)

    def train(self):
        """
        Train the CoxPH model across multiple splits and store results.
        """
        super().train()

    def get_summary_objects(self):
        """
        Summarize the concordance index (C-index) for each sample across all models
        where the sample was part of the test set in the RENT training.

        Returns
        -------
        <pandas dataframe>
            DataFrame where rows represent objects and columns represent:
            - '# test': How many times the sample was part of a test set.
            - 'mean c-index': The average C-index for the sample across all test sets.
        """
        # Convert self._target to a pandas DataFrame if it's a numpy array
        if isinstance(self._target, np.ndarray):
            self._target = pd.DataFrame(self._target, columns=['event', 'time'])

        # Initialize a DataFrame to store the summary information
        summary_df = pd.DataFrame(index=self._target.index)
        summary_df['# test'] = 0
        summary_df['mean c-index'] = 0.0

        specific_predictions = []
        for K in range(self._K):
            key = (self._best_C, self._best_l1_ratio, K)
            if key in self._predictions:
                specific_predictions.append(self._predictions[key])
            else:
                print(f"KeyError: {key} not found in predictions.")
                specific_predictions.append(None)
        
        for K, specific_prediction in enumerate(specific_predictions):
            key = (self._best_C, self._best_l1_ratio, K)
            if key not in self._test_indices:
                print(f"KeyError: {key} not found in _test_indices.")
                continue

            test_indices = self._test_indices[key]

            for count, ind in enumerate(test_indices):
                y_test = self._target.loc[ind]
                pred_risk = specific_prediction[count]

                summary_df.loc[ind, '# test'] += 1

                c_index = concordance_index_censored(
                    y_test['event'], y_test['time'], [pred_risk])[0]

                summary_df.loc[ind, 'mean c-index'] += c_index

        # Calculate mean c-index only for samples that have been tested at least once
        summary_df['mean c-index'] = summary_df.apply(
            lambda row: row['mean c-index'] / row['# test'] if row['# test'] > 0 else np.nan,
            axis=1
        )
        
        return summary_df
        # # Debugging: Print values of _best_C, _best_l1_ratio, and _K
        # print(f"Debug: _best_C = {self._best_C}, _best_l1_ratio = {self._best_l1_ratio}, _K = {self._K}")
        
        # # Debugging: Print the keys of self._predictions
        # print(f"Debug: self._predictions keys = {list(self._predictions.keys())}")

        # specific_predictions = []
        # for K in range(self._K):
        #     key = (self._best_C, self._best_l1_ratio, K)
        #     if key in self._predictions:
        #         specific_predictions.append(self._predictions[key])
        #     else:
        #         print(f"KeyError: {key} not found in predictions.")
        #         specific_predictions.append(None)
        
        # for K, specific_prediction in enumerate(specific_predictions):
        #     key = (self._best_C, self._best_l1_ratio, K)
        #     test_indices = self._test_indices[key]

        #     for count, ind in enumerate(test_indices):
        #         y_test = self._target[ind]
        #         pred_risk = specific_prediction[count]

        #         self._incorrect_labels.loc[ind, '# test'] += 1

        #         c_index = concordance_index_censored(
        #             y_test['event'], y_test['time'], [pred_risk])[0]

        #         self._incorrect_labels.loc[ind, 'mean c-index'] += c_index

        # self._incorrect_labels['mean c-index'] = (
        #     self._incorrect_labels['mean c-index'] / self._incorrect_labels['# test']
        # )
        # return self._incorrect_labels
        
        # if not hasattr(self, '_best_C') or not hasattr(self, '_best_l1_ratio'):
        #     sys.exit('Run train() first!')

        # # Initialize the summary DataFrame
        # self._incorrect_labels = pd.DataFrame({
        #     '# test': np.repeat(0, np.shape(self._data)[0]),
        #     'mean c-index': np.repeat(0.0, np.shape(self._data)[0])
        # })
        # self._incorrect_labels.index = self._indices.copy()

        # # Iterate over all folds to collect predictions
        # specific_predictions = [
        #     self._predictions[(self._best_C, self._best_l1_ratio, K)]
        #     for K in range(self._K)
        # ]

        # for K, specific_prediction in enumerate(specific_predictions):
        #     key = (self._best_C, self._best_l1_ratio, K)
        #     test_indices = self._test_indices[key]

        #     # Calculate the C-index for each sample in the test set
        #     for count, ind in enumerate(test_indices):
        #         y_test = self._target[ind]
        #         pred_risk = specific_prediction[count]

        #         # Increase the '# test' count for this sample
        #         self._incorrect_labels.loc[ind, '# test'] += 1

        #         # Calculate the concordance index for the individual sample
        #         c_index = concordance_index_censored(
        #             y_test['event'], y_test['time'], [pred_risk])[0]

        #         # Accumulate the C-index for averaging later
        #         self._incorrect_labels.loc[ind, 'mean c-index'] += c_index

        # # Calculate the mean C-index for each sample
        # self._incorrect_labels['mean c-index'] = (
        #     self._incorrect_labels['mean c-index'] / self._incorrect_labels['# test']
        # )

        # # Handle cases where a sample was not included in any test set
        # self._incorrect_labels.loc[self._incorrect_labels['# test'] == 0, :] = np.nan

        # return self._incorrect_labels