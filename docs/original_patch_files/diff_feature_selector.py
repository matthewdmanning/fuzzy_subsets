Index: dmso_model_dev / models / feature_selector.py
IDEA
additional
info:
Subsystem: com.intellij.openapi.diff.impl.patch.BaseRevisionTextPatchEP
< + >
import builtins\r\nimport
copy\r\nimport
dataclasses\r\nimport
os\r\nimport
pprint\r\nimport
itertools\r\nimport
prettyprint\r\nimport
functools\r\nimport
logging\r\nimport
typing\r\nfrom
typing
import Any, Set\r\nfrom
collections
import namedtuple\r\nimport
cachetools\r\nimport
numpy as np\r\nimport
pandas as pd\r\nimport
sklearn.utils.validation\r\nfrom
sklearn.pipeline
import Pipeline\r\nfrom
sklearn.linear_model
\r\nfrom
sklearn.feature_selection
 \\\r\n
VarianceThreshold\r\nfrom
sklearn.metrics
import balanced_accuracy_score, confusion_matrix, roc_curve, classification_report, \\\r\n
matthews_corrcoef\r\nfrom
sklearn.feature_selection
\r\nfrom
sklearn.ensemble
\r\nfrom
sklearn.preprocessing
import RobustScaler, Normalizer\r\nfrom
sklearn.base
\r\nfrom
sklearn.utils
import check_X_y\r\nfrom
statsmodels.stats.outliers_influence
import variance_inflation_factor\r\nfrom
dmso_model_dev.data_handling.padel_categorization
import load_padel_df\r\nfrom
dmso_model_dev.data_handling
import

FeatureAccessor\r\n\r\nPADEL_GROUPS_DIR = \"{}}padel/padel_desc_groups.csv\"\r\nGROUP_COL_NAMES = ('Type', 'Number', 'Descriptor', 'Class')\r\nWEIGHT_NAMES = ('mass', 'charges', 'van der Waals', 'Sanderson', 'electronegativites', 'polarizabilities', 'ionization', 'I-state', )\r\n# Model running options.\r\nlinear_dim_reduction = False\r\nplot_tsne = False\r\nplot_logreg = True\r\nforest = True\r\n'''\r\n@dataclasses.dataclass\r\nclass TrainingOpts:\r\n    run_debug: bool\r\n    corr_meth: str\r\n    sample_wts: pd.Series\r\n    pjobs: int\r\n    data_dir: str | os.PathLike\r\n    stats_dir: str | os.PathLike\r\n'''\r\ndef make_training_opts(opts):\r\n    options = dataclasses.make_dataclass('options', ((k, type(v)) for k, v in opts.items()))(**opts)\r\n    options.__module__ = __name__\r\n    return options\r\n\r\n# TODO: Check assumptions/requirements of estimators.\r\nclass FeatureSelectorTrainer:\r\n    _group_names = ('zero_var', 'nonzero_var', 'sparse', 'dense', 'discrete', 'cont', 'nonneg', 'neg')\r\n\r\n    def __init__(self, data, labels=None, fs=None, options=None, logger=None, subsets=None, *args, **kwargs):\r\n        self.options = self.set_options(options, log_opts=logger)\r\n        if self.options.run_debug:\r\n            self.X = FeatureFrame(df=data.sample(n=2500, axis='index').sample(n=100, axis='columns').copy(deep=True), options=self.options)\r\n            self.options.sample_wts = self.options.sample_wts[self.X.feat_frame.index]\r\n        elif fs and type(fs) is FeatureFrame:\r\n            self.X = fs\r\n        else:\r\n            self._X = FeatureFrame(df=data.copy(deep=True), options=self.options)\r\n        self._y = labels\r\n        self._corr = None\r\n        self._corr_mat = None\r\n        self._condition_num = None\r\n        self._descriptor_groups_df = None\r\n        self._calc_values()\r\n        # self.feature_properties = dict()\r\n        # self.feat_scorer_types = dict()\r\n        self._scalers = dict()\r\n        # self._get_desc_groups(grouping_dir=PADEL_GROUPS_DIR, cols=GROUP_COL_NAMES)\r\n        # x_sample = train_X.sample(axis=1, frac=0.2, random_state=0)\r\n        if not debug:\r\n            if len(self.X.sparse) < MIN_NUM_FEATS['sparse']:\r\n                logger.warning('Too few sparse features: \\n{}'.format(self.X.sparse))\r\n                raise ValueError\r\n            if len(self.X.dense) < MIN_NUM_FEATS['dense']:\r\n                logger.warning('Too few dense features: \\n{}'.format(self.X.dense))\r\n                raise ValueError\r\n        cweights = self.y.copy().replace({1: 1, 0: 10})\r\n        # self.__initialize_data_history__()\r\n\r\n    '''\r\n    def __initialize_data_history__(self):\r\n        for a in self.data_dict:\r\n            self.data_history[a.__name] = [a]\r\n    '''\r\n\r\n    # Set default values for options named tuple.\r\n    def set_options(self, f_opts, log_opts=None):\r\n        global logger\r\n        logging.basicConfig(**log_opts)\r\n        logger = logging.getLogger(name='logger')\r\n        if f_opts['run_debug']:\r\n            logger.setLevel(logging.DEBUG)\r\n        else:\r\n            logger.setLevel(logging.INFO)\r\n        dc = make_training_opts(f_opts)\r\n        return dc\r\n\r\n    def _calc_values(self):\r\n        self._corr = self.X.feat_frame[self.X.nonzero_vars].corrwith(other=self.y, axis='columns',method=self.options.corr_meth)\r\n        # self._set_descriptor_groups_df()\r\n\r\n    def check_feat_num(self):\r\n        for k, v in self.options.asdict().items():\r\n            if 'min' in k and 'feats' in k and len(X.sparse.tolist()) < v:\r\n                logger.error('Number of {} features is smaller than minimum of {}'.format(k, v))\r\n\r\n    @property\r\n    def corr_mat(self):\r\n        return self._corr_mat\r\n\r\n\r\n    # aweights is used to balance covariance estimates for different subsets of observations.\r\n    def calc_cov_mat(self, calc_cov=None, filter_nan=False, sample_wts='auto', *args, **kwargs):\r\n        if calc_cov is not None:\r\n            self._corr_mat = calc_cov\r\n        cov_path = '{}cov.pkl'.format(self.options.stats_dir)\r\n        if type(self.corr_mat) is pd.DataFrame and not self.corr_mat.empty and self.corr_mat.shape[1] == self.X.feat_frame.shape[1]:\r\n            logger.info('Covariance matrix has already been calculated.')\r\n            return self.corr_mat\r\n        if sample_wts == 'auto':\r\n            sample_wts = self.options.sample_wts\r\n        # _corr = self.X.feat_frame[self.X.dense].corr(method=self.options.corr_meth)\r\n        dense_ind = self.X.feat_frame.index.intersection(self.X.dense)\r\n        # print(self.X.feat_frame[self.X.dense].shape, sample_wts.shape)\r\n        _corr = pd.DataFrame(np.cov(self.X.feat_frame, rowvar=False, aweights=sample_wts, ddof=0), index=self.X.feat_frame.columns\r\n                             , columns=self.X.feat_frame.columns)\r\n        assert _corr is not None\r\n        logging.info('Covariance Matrix:\\n{}'.format(_corr))\r\n        logging.info('CoV NaNs: {}'.format(_corr.isna().astype(int).sum().sum()))\r\n        if filter_nan:\r\n            # Remove features with no non-NaN correlations.\r\n            na_corr = _corr.isna().astype(int)\r\n            sum_na = na_corr.sum(axis=0)\r\n            while na_corr[sum_na > 0].size > 0:\r\n                col_na = na_corr[sum_na == 0]\r\n                most_na = na_corr.idxmax()\r\n                logger.info('NA value for cols: {}'.format(col_na))\r\n                # logger.warning('These features have no valid correlations: {}'.format(col_na))\r\n                _corr.drop(columns=col_na, inplace=True).drop(index=col_na, inplace=True)\r\n                na_corr = _corr.isna().astype(int)\r\n                sum_na = na_corr.sum(axis=0)\r\n            # logger.info('Correlation matrix: {}'.format(_corr))\r\n        assert _corr is not None\r\n        logger.debug('Covariance matrix with weighting of {}:\\n{}'.format(self.options.sample_wts, _corr))\r\n        self._corr_mat = _corr\r\n        return _corr\r\n\r\n    @cachetools.cached(cache={})\r\n    def set_cov_pairs(self, cov_thresh=None):\r\n        if not cov_thresh:\r\n            cov_thresh = 0.9\r\n        corr_mat = self.X.corr_mat\r\n        corr_dict = corr_mat.to_dict(orient='series')\r\n        same_list, close_list = list(), list()\r\n        for ind, col in itertools.combinations(corr_dict.keys(), r=2):\r\n            val = corr_mat.loc[ind, col]\r\n            print(ind, loc, val)\r\n            if abs(val) == 1:\r\n                same_list.append((col, ind))\r\n                logger.info('{} and {} with R of {}'.format(ind, col, val))\r\n            elif abs(val) > cov_thresh:\r\n                logger.info('{} and {} are highly correlated with an R of {}'.format(ind, col, val))\r\n                close_list.append([(ind, col), val])\r\n        logger.info(close_list)\r\n        self.X._corr_pairs = same_list\r\n\r\n    @cachetools.cached(cache={})\r\n    def set_condition_num(self):\r\n        if type(self.condition_num) is pd.DataFrame and not self.condition_num.empty:\r\n            logger.warning('Condition number is already calculated for this FeatureFrame.')\r\n            return self.condition_num\r\n        # Eigenvalue method for collinearity.\r\n        eigenvalues = np.linalg.eigvals(self.corr_mat)\r\n        condition_index = np.sqrt(np.max(eigenvalues) / eigenvalues)\r\n        pprint.PrettyPrinter().pprint(condition_index)\r\n        logger.info(f\"Condition Index: {condition_index}\")\r\n        np.save('{}condition_index'.format(self.options.stats_dir), arr=condition_index)\r\n        condition_index = pd.DataFrame(condition_index)\r\n        if condition_index.shape[0] == self.corr_mat.shape[0]:\r\n            condition_index.reindex_like(self.corr_mat.index)\r\n        if condition_index.shape[1] == self.corr_mat.shape[1]:\r\n            condition_index.columns.rename(self.corr_mat.columns)\r\n        self._condition_num = condition_index\r\n\r\n    @cachetools.cached(cache={})\r\n    def multi_collin_feats(self):\r\n        dense_df = self.X.feat_frame[self.X.dense].copy(deep=True)\r\n        logger.debug(self.X.dense)\r\n        # sklearn.utils.assert_all_finite(dense_df)\r\n        vif = pd.DataFrame()\r\n        vif[\"features\"] = dense_df.columns\r\n        vif[\"VIF Factor\"] = [variance_inflation_factor(np.array(dense_df.values.tolist(), dtype=float), i) for i in\r\n                             range(dense_df.shape[1])]\r\n        vif.set_index(keys='features', drop=True, inplace=True)\r\n        logger.info(vif)\r\n        return vif\r\n\r\n    def seq_linear_select(self, df=None, label=None, method='auto', seq_linear_select_scoring='balanced_accuracy',\r\n                          num_parallel=1):\r\n        if not df:\r\n            df = self.X\r\n        if not label:\r\n            label = self.y\r\n        check_X_y(df, label)\r\n        if method == 'elastic' or (method == 'auto' and df.shape[1] < 50):\r\n            seq_linear_select_estimator = ElasticNetCV(n_jobs=num_parallel, random_state=0, max_iter=5000, tol=1e-5,\r\n                                                       cv=3)\r\n        else:\r\n            seq_linear_select_estimator = LassoCV(fit_intercept=False, max_iter=2500, tol=1e-5, n_jobs=num_parallel,\r\n                                                  random_state=0,\r\n                                                  cv='balanced')\r\n        seq_linear_selector = SequentialFeatureSelector(estimator=seq_linear_select_estimator,\r\n                                                        scoring=seq_linear_select_scoring,\r\n                                                        n_jobs=num_parallel)\r\n        return seq_linear_selector.fit(df)\r\n\r\n    # TODO: Delete groups with no members or just one member.\r\n    def _set_descriptor_groups_df(self, grouping_dir=PADEL_GROUPS_DIR, cols=GROUP_COL_NAMES, use3d=False):\r\n        padel_name_df = load_padel_df()\r\n        short_long_zip = zip(padel_name_df['Descriptor name'].tolist(), padel_name_df['Description'].tolist())\r\n        short_long_dict = dict([(a, b) for a, b in short_long_zip])\r\n        if grouping_dir:\r\n            desc_groups_df = pd.read_csv(filepath_or_buffer=grouping_dir, usecols=cols)\r\n            # desc_groups_df.dropna(subset='Descriptor', inplace=True)\r\n            if not use3d:\r\n                # logging.info(desc_groups_df['Descriptor'])\r\n                desc_groups_df.drop(desc_groups_df[desc_groups_df['Class'] == '3D'].index, inplace=True)\r\n            long_dict = dict()\r\n            for _, i, desc_group in desc_groups_df[desc_groups_df['Number'] > 1][['Type', 'Descriptor']].itertuples():\r\n                # logging.warning('{} {}'.format(type(desc_group), desc_group))\r\n                # i, desc_group = vals.values\r\n                ind = str(i)\r\n                logging.info('Desc row: {}: {}'.format(ind, desc_group))\r\n                if desc_group == 'nan' or desc_group == np.NaN or not desc_group or type(desc_group) is float:\r\n                    long_dict.update([(ind, list())])\r\n                    continue\r\n                elif type(desc_group) is str:\r\n                    if ',' not in desc_group:\r\n                        if desc_group in short_long_dict.keys():\r\n                            long_dict.update([(ind, list(desc_group))])\r\n                            continue\r\n                        else:\r\n                            logging.error('COULD NOT FIND DESCRIPTOR IN LIST: {}'.format(desc_group))\r\n                            continue\r\n                    elif ',' in desc_group:\r\n                        key_list = [d.strip() for d in desc_group.split(',')]\r\n                        desc_list = list()\r\n                        if any([len(k) == 1 for k in key_list]):\r\n                            logging.warning('SPLITTING INTO SINGLE CHARACTERS!')\r\n                        else:\r\n                            for d in key_list:\r\n                                if len(d) <= 1:\r\n                                    logging.error('Descriptor splitting for {} gives single characters'.format(key_list))\r\n                                elif d not in short_long_dict.keys():\r\n                                    logging.warning('Descriptor not found in key list!!!: \"{}\"'.format(d))\r\n                                    continue\r\n                                    # key_list.remove(d)\r\n                                else:\r\n                                    desc_list.append((d, short_long_dict[d]))\r\n                                    logger.debug('{} in keys!'.format(d))\r\n                            long_dict = dict(desc_list)\r\n                            if len(desc_list) == 0:\r\n                                logging.warning('Empty descriptor group label: {}: {}'.format(ind, desc_group.split(',')))\r\n                                long_dict.update({ind: list()})\r\n                            elif len(desc_list) == 0 and any([k in desc_group.strip(',') for k in short_long_dict.keys()]):\r\n                                logging.error('DESCRIPTOR LIST NOT BEING SPLIT CORRECTLY! \\n{}VS.\\n{}'.format(desc_group, desc_list))\r\n                                raise ValueError\r\n                            else:\r\n                                long_dict.update({ind: list()})\r\n                    else:\r\n                        logging.error('Desc list {} is string but is not in descriptor DF and does not contain comma delimiter.'.format(key_list))\r\n                        raise ValueError\r\n                elif type(desc_group) is list:\r\n                    long_dict.update((ind, desc_group))\r\n                elif type(desc_group) is tuple or type(desc_group) is set:\r\n                    long_dict.update((ind, list(desc_group)))\r\n                else:\r\n                    logging.error('Unknown type {} for descriptor group {}'.format(type(desc_group), desc_group))\r\n                    raise TypeError\r\n                    # long_dict.update([(ind, [short_long_dict[str(d)] for d in desc_group.split(',') if str(d) in short_long_dict.keys()])])\r\n            logger.info('Input dictionary: {}'.format(long_dict.items()))\r\n            for k, v in long_dict.items():\r\n                if type(v) is not list:\r\n                    long_dict[k] = list(v)\r\n            logger.info('Input dictionary: {}'.format(long_dict.items()))\r\n            long_df = pd.DataFrame.from_dict(long_dict)\r\n            if long_df.empty:\r\n                long_df = pd.DataFrame.from_dict(long_dict, orient='index')\r\n            if long_df.empty:\r\n                long_df = pd.DataFrame.from_records(data=long_dict)\r\n            if long_df.empty:\r\n                long_df = pd.Series(data=long_dict.values(), index=long_dict.keys(), name='Long')\r\n            if long_df.empty:\r\n                logger.warning('Long descriptor DF is empty!')\r\n                raise ValueError\r\n            if type(long_df) is pd.DataFrame and len(long_df.shape) > 1 and long_df.shape[0] < long_df.shape[1]:\r\n                long_df = long_df.T\r\n            else:\r\n                logger.info(type(long_df))\r\n            logger.info('Brand new long_df: {}'.format(long_df))\r\n            try:\r\n                long_df.rename(name=['Long'], inplace=True)\r\n            except TypeError:\r\n                logger.error('Could not rename long_df')\r\n            logger.info('Long DF: \\n{}'.format(long_df.to_string()))\r\n            new_desc_df = desc_groups_df.set_index(keys='Type').join(long_df, how='inner')\r\n            if new_desc_df.empty:\r\n                new_desc_df = desc_groups_df.join(long_df.reset_index(), how='inner', ignore_index=True)\r\n            # desc_groups_df['Long'] = desc_groups_df['Descriptor'].apply(func=lambda x: [[short_long_dict[a.strip()] for a in d if a != 'NaN' and a != np.NaN] for d.split(',') in x if len(d) > 1 else x])\r\n            # desc_groups_df['Long'] = desc_groups_df['Descriptor'].apply(func=lambda x: [short_long_dict[d.rstrip(',')] for d in x.split()])\r\n            logger.info('Col Names: {}'.format(new_desc_df.columns))\r\n            logger.info('Long Name Descriptors: \\n{}'.format(new_desc_df))\r\n            # new_desc_df.sort_values(key=lambda x: len(x), inplace=True, ascending=False)\r\n        else:\r\n            raise FileNotFoundError\r\n        self._descriptor_groups_df = new_desc_df\r\n\r\n    @cachetools.cached(cache={})\r\n    def scale_df(self, df=None, scale_to=None, select_scalers='all', *args, **kwargs):\r\n        df_scaled = dict()\r\n        if not df:\r\n            df = self.X.cont\r\n        if not self.scalers:\r\n            self.scalers = self.X.scalers(scale_to, *args, **kwargs)\r\n        if select_scalers != 'all':\r\n            select_scalars = [key for key in self.scalers.keys() if key in select_scalers]\r\n        else:\r\n            select_scalers = self.scalers\r\n        for name, scaler in self.scalers.items():\r\n            if sklearn.utils.validation.check_is_fitted(scaler):\r\n                df_scaled.update((name, scaler.transform(df)))\r\n            else:\r\n                if scale_to:\r\n                    scaler.fit(scale_to)\r\n                    scaler.transform(df)\r\n                else:\r\n                    scaler.fit_transform(df)\r\n                self.scalers.update((name, self.scalers))\r\n        return df_scaled\r\n\r\n    # Set Sample weight option.\r\n    def isolate_observations(self, fit_to=None, sample_wts='auto', contamination=0.001, rstate=None, n_jobs=-1):\r\n        from sklearn.ensemble import IsolationForest\r\n        if not fit_to:\r\n            fit_to = self.X.feat_frame[self.X.nonzero_vars]\r\n        if sample_wts == 'auto':\r\n            sample_wts = None\r\n        iso = IsolationForest(n_jobs=n_jobs, contamination=contamination, random_state=0)\r\n        if fit_to:\r\n            iso = iso.fit(fit_to, sample_weight=sample_wts)\r\n        return iso\r\n\r\n    @property\r\n    def X(self):\r\n        return self._X\r\n\r\n    @property\r\n    def y(self):\r\n        return self._y\r\n\r\n    @staticmethod\r\n    def get_high_vars(self, n=50):\r\n        scaler = RobustScaler()\r\n        high_thresh = VarianceThreshold()\r\n        high_thresh_pipe = Pipeline(steps=[('high_var_scaler', scaler), ('high_thresh', high_thresh)])\r\n        return high_thresh_pipe\r\n\r\n    @property\r\n    def corr(self):\r\n        return self._corr\r\n\r\n    @property\r\n    def descriptor_groups_df(self):\r\n        return self._descriptor_groups_df\r\n\r\n    @descriptor_groups_df.setter\r\n    def descriptor_groups_df(self, value):\r\n        if self._descriptor_groups_df.shape[0] > 1:\r\n            pass\r\n        self._descriptor_groups_df = value\r\n\r\n    @property\r\n    def scalers(self):\r\n        return self._scalers\r\n\r\n    @cachetools.cached(cache={})\r\n    @scalers.setter\r\n    def scalers(self, fit_to=None, robust_iqr=(0.05, 0.95), unit_robust_iqr=(0.05, 0.95), normalizer_norm='l2', *args,\r\n                **kwargs):\r\n        \"\"\"\r\n        :param fit_to:\r\n        :param robust_iqr:\r\n        :param unit_robust_iqr:\r\n        :param normalizer_norm:\r\n        :param args:\r\n        :param kwargs:\r\n        :return:\r\n        \"\"\"\r\n\r\n        # noinspection PyArgumentEqualDefault\r\n        scaler_dict = dict([\r\n            ('robust', RobustScaler(quantile_range=robust_iqr, unit_variance=False, **kwargs)),\r\n            ('robust_unit', RobustScaler(quantile_range=unit_robust_iqr, unit_variance=True, **kwargs)),\r\n            ('normal', Normalizer(norm=normalizer_norm, **kwargs))])\r\n        if fit_to:\r\n            [val.fit(fit_to.to_numpy()) for val in scaler_dict.values()]\r\n        self._scalers = scaler_dict\r\n\r\ndef iter_feats(feat_groups):\r\n    for name, feat_dict in feat_groups:\r\n        col_group, selector = feat_dict\r\n\r\n\r\nREQUIRED_OPTS = ('tol_discrete', 'tol_sparse', 'sparsity')\r\n\r\n\r\nclass FeatureFrame:\r\n\r\n    def __init__(self, df=None, options=None, *args, **kwargs):\r\n        self._feat_frame = df\r\n        self._original = self.feat_frame.copy(deep=True)\r\n        self._options = options\r\n        self._sparse = None\r\n        self._dense = None\r\n        self._discrete = None\r\n        self._cont = None\r\n        self._neg = None\r\n        self._nonneg = None\r\n        self._feat_vars = None\r\n        self._feat_stat_scorer = None\r\n        self._corr_pairs = None\r\n        self._corr_mat = None\r\n        self._set_attr()\r\n\r\n    @property\r\n    def feat_frame(self):\r\n        return self._feat_frame\r\n\r\n    @property\r\n    def original(self):\r\n        return self._original\r\n\r\n    @property\r\n    def options(self):\r\n        return self._options\r\n\r\n    @options.setter\r\n    def options(self, value):\r\n        self._options = format_options(value)\r\n\r\n    def _set_attr(self):\r\n        self._set_feat_vars()\r\n        self._set_sparse_dense()\r\n        self._set_discrete_cont()\r\n        self._set_neg_nonneg()\r\n        self._set_corr_mat()\r\n        # self.scalers()\r\n        self._set_feat_stat_scorer(None)\r\n\r\n    @property\r\n    def feat_vars(self):\r\n        return self._feat_vars\r\n\r\n    def _set_feat_vars(self, skipna=True, numeric_only=False, *args, **kwargs):\r\n        feat_vars = self.original.var(axis='index', skipna=skipna, numeric_only=numeric_only)\r\n        logger.info('Feature variances: {}'.format(feat_vars))\r\n        self._feat_vars = feat_vars\r\n        if feat_vars.empty or feat_vars.dropna().empty:\r\n            raise ValueError\r\n        return feat_vars\r\n        # self.nonzero_vars((self.feat_vars > 0).index.append(pd.Index([c for c in self.original.columns if c not in self.feat_vars.index])))\r\n        # self.zero_vars(self.original.columns.symmetric_difference(self.nonzero_vars))\r\n\r\n    @property\r\n    def desc_groups(self):\r\n        return self._desc_groups\r\n\r\n    @property\r\n    def feat_stat_scorer(self):\r\n        return self._feat_stat_scorer\r\n\r\n    def _set_feat_stat_scorer(self, value):\r\n        cont_mi_clf = functools.partialmethod(mutual_info_classif, discrete_features=False)\r\n        disc_mi_clf = functools.partialmethod(mutual_info_classif, discrete_features=True)\r\n        return dict(\r\n            [(f_classif, 'real'), (cont_mi_clf, 'real'), (disc_mi_clf, 'discrete'), (chi2, 'nonneg')])\r\n\r\n    def _set_neg_nonneg(self):\r\n        nonneg_cols, neg_cols = list(), list()\r\n        for col, ser in self.feat_frame.items():\r\n            if (ser.values < 0).any():\r\n                neg_cols.append(col)\r\n            else:\r\n                nonneg_cols.append(col)\r\n        self.nonneg = pd.Index(nonneg_cols)\r\n        self.neg = pd.Index(neg_cols)\r\n        logger.debug(self.neg)\r\n        logger.debug(self.nonneg)\r\n\r\n    @property\r\n    def discrete(self):\r\n        return self._discrete\r\n\r\n    @discrete.setter\r\n    def discrete(self, value):\r\n        if type(value) is not pd.Index:\r\n            raise AttributeError\r\n        self._discrete = value\r\n\r\n    @property\r\n    def cont(self):\r\n        return self._cont\r\n\r\n    @cont.setter\r\n    def cont(self, value):\r\n        if type(value) is not pd.Index:\r\n            raise AttributeError\r\n        self._cont = value\r\n\r\n    def _set_discrete_cont(self):\r\n        nonzeros = self.feat_frame[self.nonzero_vars].copy()\r\n        remainder = nonzeros.round(0).sub(nonzeros).abs()\r\n        logger.debug(remainder)\r\n        self.discrete = nonzeros.columns[(remainder < self.options.tol_discrete).all()]\r\n        self.cont = nonzeros.columns.symmetric_difference(self.discrete)\r\n        # logger.info('Discrete: {}', self.discrete)\r\n        # logger.info('Continuous: {}', self.cont)\r\n\r\n    @property\r\n    def corr_mat(self):\r\n        return self._corr_mat\r\n\r\n    # aweights is used to balance covariance estimates for different subsets of observations.\r\n    def _set_corr_mat(self, corr_method='pearson', filter_nan=False, *args, **kwargs):\r\n        _corr = self.feat_frame[self.dense].corr(method=corr_method)\r\n        np.cov()\r\n        if filter_nan:\r\n            # Remove features with no non-NaN correlations.\r\n            na_corr = _corr.isna().astype(int)\r\n            col_na = _corr[na_corr.sum(axis=0) < _corr.shape[1] - 1]\r\n            logger.info('Correlation matrix: {}'.format(_corr))\r\n            logger.info('NA value for cols: {}'.format(col_na))\r\n            logger.warning('These features have no valid correlations: {}'.format(col_na))\r\n            corr_matrix = _corr.loc[col_na, col_na]\r\n        else:\r\n            corr_matrix = _corr\r\n        logger.debug(corr_matrix)\r\n        self._corr_mat = corr_matrix\r\n\r\n    @property\r\n    def corr_pairs(self):\r\n        return self._corr_pairs\r\n\r\n    @corr_pairs.setter\r\n    def corr_pairs(self, corr_method='kendall', *args, **kwargs):\r\n        corr_val = self.feat_frame.corr_mat(corr_method=corr_method, *args, **kwargs)\r\n        logger.info(corr_val, corr_val.shape)\r\n        corr_sorted = corr_val.unstack().sort_values(kind='quicksort', ascending=False)\r\n        logging.info('Sorted Correlation values:\\n{}'.format(corr_sorted[:10]))\r\n        corr_set = set()\r\n        for i in range(0, self.X.dense.shape[1]):\r\n            for j in range(0, i + 1):\r\n                corr_set.add((self.X.dense.columns[i], self.X.dense.columns[j]))\r\n        self._corr_pairs = corr_set\r\n\r\n    @property\r\n    def sparse(self):\r\n        return self._sparse\r\n\r\n    @sparse.setter\r\n    def sparse(self, value):\r\n        if type(value) is not pd.Index:\r\n            raise AttributeError\r\n        self._sparse = value\r\n\r\n    @property\r\n    def zero_vars(self):\r\n        return self._zero_vars\r\n\r\n    @zero_vars.setter\r\n    def zero_vars(self, value):\r\n        if type(value) is not pd.Index:\r\n            raise AttributeError\r\n        self._zero_vars = value\r\n\r\n    @property\r\n    def nonzero_vars(self):\r\n        return self._nonzero_vars\r\n\r\n    @nonzero_vars.setter\r\n    def nonzero_vars(self, value):\r\n        if type(value) is not pd.Index:\r\n            raise AttributeError\r\n        self._nonzero_vars = value\r\n\r\n    @cachetools.cached(cache={})\r\n    def _set_sparse_dense(self, all_freq=True, ignore_nan=False):\r\n        if 0 < self.options.tol_sparse < 1:\r\n            freq_cut = 1 - self.options.tol_sparse\r\n        elif 1 < self.options.tol_sparse < self.X.vary.shape[0]:\r\n            freq_cut = self.options.tol_sparse / self.X.vary.shape[0]\r\n        else:\r\n            raise ValueError\r\n        logger.info('Sparse tolerance: {}'.format(self.options.tol_sparse))\r\n        logger.info('Freq cut: {}'.format(freq_cut))\r\n        sparse_list, dense_list, zero_list, freq_dict = list(), list(), list(), dict()\r\n        for col, ser in self._pd_feat_freq(ignore_nan=ignore_nan):\r\n            if all_freq:\r\n                freq_dict[col] = ser\r\n            sermax = ser.max(skipna=ignore_nan)\r\n            # print('Feature freq max: {}'.format(sermax))\r\n            logging.debug('Feature maximum: {}'.format(sermax))\r\n            if freq_cut < sermax < 1.:\r\n                sparse_list.append(col)\r\n            elif freq_cut > sermax:\r\n                dense_list.append(col)\r\n            else:\r\n                zero_list.append(col)\r\n        self.sparse = pd.Index(sparse_list)\r\n        self.dense = pd.Index(dense_list)\r\n        nonzeros = sparse_list.copy()\r\n        nonzeros.extend(dense_list)\r\n        self.nonzero_vars = pd.Index(nonzeros)\r\n        self.zero_vars = pd.Index(zero_list)\r\n        # self.feature_properties['zero_vars'] = self.X.vary.columns.symmetric_difference(pd.Index(data=nonzeros))\r\n\r\n    @cachetools.cached(cache={})\r\n    def _pd_feat_freq(self, ignore_nan=False):\r\n        # logger.info('Feature variances: ', self.feat_vars)\r\n        for col, ser in self.original.items():\r\n            yield col, ser.value_counts(normalize=True, dropna=ignore_nan)\r\n\r\n\r\n@functools.singledispatch\r\ndef format_options(arg):\r\n    self._options = args\r\n\r\n\r\n@format_options.register(dict)\r\ndef _(arg):\r\n    for o in REQUIRED_OPTS:\r\n        if o not in arg.keys():\r\n            logger.error('Required option {} not found.'.format(o))\r\n            raise KeyError\r\n    return arg\r\n\r\n# selector = sklearn.feature_selection.SelectFromModel(estimator=score_f, max_features=n_feats, prefit=True, threshold='median')\r\n\r\n# padel_groups = _get_desc_groups(X, grouping_dir=PADEL_GROUPS_DIR, cols=GROUP_COL_NAMES)\r\n# feat_group_names = ['Dense', 'Matrix-based', 'E-States', 'Sparse', 'All Features']\r\n# feat_selector_bool = [False, True, True, True, False]\r\n# feat_col_list = [dense, matrix, estate, sparse,  X_train.columns]\r\n# zip(feat_group_names, feat_col_list)\r\n# feat_col_list = [sparse, dense, matrix, estate, X_train.columns]\r\n\r\n\r\n# train_feats = X_train.copy()\r\n# train_labels = train_y.copy()\r\n
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
< + > UTF - 8
== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == =
diff - -git
a / dmso_model_dev / models / feature_selector.py
b / dmso_model_dev / models / feature_selector.py
--- a / dmso_model_dev / models / feature_selector.py
+++ b / dmso_model_dev / models / feature_selector.py


@ @-53

, 6 + 53, 7 @ @
return options

# TODO: Check assumptions/requirements of estimators.
+  # TODO: Alter when *sample* is run data, to after zero-var drop and before cov calculations.


class FeatureSelectorTrainer:
    _group_names = ('zero_var', 'nonzero_var', 'sparse', 'dense', 'discrete', 'cont', 'nonneg', 'neg')


@ @-64

, 8 + 65, 9 @ @
elif fs and type(fs) is FeatureFrame:
self.X = fs
else:
-            self._X = FeatureFrame(df=data.copy(deep=True), options=self.options)
-        self.y = labels
+            self.X = FeatureFrame(df=data.copy(deep=True), options=self.options)
+  # self._X.feat_frame = self._X.feat_frame[self.X.nonzero_vars]
+        self.y = labels[self.X.feat_frame.index].astype(int)
self.corr = None
self._cov_mat = None
self._condition_num = None


@ @-76

, 6 + 78, 8 @ @
self._scalers = dict()
# self._get_desc_groups(grouping_dir=PADEL_GROUPS_DIR, cols=GROUP_COL_NAMES)
# x_sample = train_X.sample(axis=1, frac=0.2, random_state=0)
+  # self.check_feat_num()
+        '''
         if not debug:
             if len(self.X.sparse) < MIN_NUM_FEATS['sparse']:
                 logger.warning('Too few sparse features: \n{}'.format(self.X.sparse))
@@ -83,7 +87,7 @@
             if len(self.X.dense) < MIN_NUM_FEATS['dense']:
                 logger.warning('Too few dense features: \n{}'.format(self.X.dense))
                 raise ValueError
-        cweights = self.y.copy().replace({1: 1, 0: 10})
+        '''
# self.__initialize_data_history__()

'''
@@ -105,7 +109,8 @@
    return dc

def _calc_values(self):
-        self._corr = self.X.feat_frame[self.X.nonzero_vars].corrwith(other=self.y, axis='columns',method=self.options.corr_meth)
+        nonzeros = self.X.feat_frame[self.X.nonzero_vars]
+        self._corr = nonzeros.corrwith(other=self.y, axis='columns',method=self.options.corr_meth)
    # self._set_descriptor_groups_df()

def check_feat_num(self):
@@ -117,6 +122,9 @@
def corr_mat(self):
    return self._corr_mat

+    @property
+    def condition_num(self):
+        return self._condition_num

# aweights is used to balance covariance estimates for different subsets of observations.
def calc_cov_mat(self, calc_cov=None, filter_nan=False, sample_wts='auto', *args, **kwargs):
@@ -157,21 +165,25 @@
@cachetools.cached(cache={})
def set_cov_pairs(self, cov_thresh=None):
    if not cov_thresh:
-            cov_thresh = 0.9
-        corr_mat = self.X.corr_mat
-        corr_dict = corr_mat.to_dict(orient='series')
+            self.options.cov_thresh
+        if (type(self.corr_mat) is not pd.DataFrame or self.corr_mat.empty) and type(self.corr_mat) is not np.ndarray:
+            self.calc_cov_mat()
+        # corr_dict = self.corr_mat.to_dict()
    same_list, close_list = list(), list()
-        for ind, col in itertools.combinations(corr_dict.keys(), r=2):
-            val = corr_mat.loc[ind, col]
-            print(ind, loc, val)
-            if abs(val) == 1:
-                same_list.append((col, ind))
-                logger.info('{} and {} with R of {}'.format(ind, col, val))
-            elif abs(val) > cov_thresh:
-                logger.info('{} and {} are highly correlated with an R of {}'.format(ind, col, val))
-                close_list.append([(ind, col), val])
-        logger.info(close_list)
-        self.X._corr_pairs = same_list
+        for ind, col in itertools.combinations(self.corr_mat.columns.to_list(), r=2):
+            val = self.corr_mat.loc[ind, col]
+            # print(ind, col, val)
+            if np.isscalar(val):
+                if abs(val) == 1:
+                    same_list.append((col, ind))
+                    logger.info('{} and {} with R of {}'.format(ind, col, val))
+                elif abs(val) > cov_thresh:
+                    logger.info('{} and {} are highly correlated with an R of {}'.format(ind, col, val))
+                    close_list.append([(ind, col), val])
+            else:
+                logger.warning('Value in covariance matrix is not a scalar?!?\n{}'.format(val))
+        self.corr_pairs = close_list
+        return close_list

@cachetools.cached(cache={})
def set_condition_num(self):
@@ -192,36 +204,65 @@
    self._condition_num = condition_index

@cachetools.cached(cache={})
-    def multi_collin_feats(self):
-        dense_df = self.X.feat_frame[self.X.dense].copy(deep=True)
-        logger.debug(self.X.dense)
-        # sklearn.utils.assert_all_finite(dense_df)
+    def multi_collin_feats(self, subset=None):
+        if not subset:
+            subset = self.X.dense
+        subset_frame = self.X.feat_frame[subset].copy(deep=True)
+        logger.debug(subset)
+        # sklearn.utils.assert_all_finite(subset_frame)
    vif = pd.DataFrame()
-        vif["features"] = dense_df.columns
-        vif["VIF Factor"] = [variance_inflation_factor(np.array(dense_df.values.tolist(), dtype=float), i) for i in
-                             range(dense_df.shape[1])]
-        vif.set_index(keys='features', drop=True, inplace=True)
-        logger.info(vif)
+        vif["features"] = subset_frame.columns.copy(deep=True)
+        vif["VIF Factor"] = [variance_inflation_factor(np.array(subset_frame.values.tolist(), dtype=float), i) for i in
+                             range(subset_frame.shape[1])]
+        vif.sort_values(by='VIF Factor').set_index(keys='features', drop=True, inplace=True)
+        logger.info('Variance Inflation Factor:\n{}'.format(vif.to_string()))
    return vif

-    def seq_linear_select(self, df=None, label=None, method='auto', seq_linear_select_scoring='balanced_accuracy',
-                          num_parallel=1):
-        if not df:
-            df = self.X
-        if not label:
+    def seq_linear_select(self, df=None, label=None, subset='auto', method='auto', seq_linear_select_scoring='balanced_accuracy', n_feats='auto'):
+        if type(df) is not pd.DataFrame:
+            df = self.X.feat_frame[self.X.dense]
+        if type(label) is not pd.Series and type(label) is not pd.DataFrame:
        label = self.y
    check_X_y(df, label)
+        from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
+
    if method == 'elastic' or (method == 'auto' and df.shape[1] < 50):
-            seq_linear_select_estimator = ElasticNetCV(n_jobs=num_parallel, random_state=0, max_iter=5000, tol=1e-5,
+            seq_linear_select_estimator = ElasticNetCV(n_jobs=self.options.pjobs, random_state=0, max_iter=5000, tol=1e-5,
                                                   cv=3)
-        else:
-            seq_linear_select_estimator = LassoCV(fit_intercept=False, max_iter=2500, tol=1e-5, n_jobs=num_parallel,
+        elif type(seq_linear_select_scoring) is str and 'lasso' in seq_linear_select_scoring.lower():
+            seq_linear_select_estimator = LassoCV(fit_intercept=False, max_iter=2500, tol=1e-5, n_jobs=self.options.pjobs,
                                              random_state=0,
                                              cv='balanced')
-        seq_linear_selector = SequentialFeatureSelector(estimator=seq_linear_select_estimator,
+        else:
+            seq_linear_select_estimator = method
+        seq_feat_selector = SequentialFeatureSelector(estimator=seq_linear_select_estimator,
                                                    scoring=seq_linear_select_scoring,
-                                                        n_jobs=num_parallel)
-        return seq_linear_selector.fit(df)
+                                                        n_jobs=self.options.pjobs, n_features_to_select=n_feats)
+        return seq_feat_selector
+
+
+    def custom_recursive_eliminator(self,  n_drop):
+        from sklearn.feature_selection import RFE, RFECV
+        from sklearn.feature_selection import mutual_info_classif
+        dropped_feats = list()
+        # First pass
+        vif = self.multi_collin_feats()
+        multicos = vif[vif >= 5.0]
+        dropped_feats.append(multicos.index.tolist())
+        feat_list = list()
+        cov_list = [(i, j, self.corr_mat.to_dict()[j][i]) for i, j in self.X.corr_pairs]
+        [feat_list.extend([i, j]) for i, j in self.X.corr_pairs]
+        feat_counts = pd.Series(feat_list).value_counts()
+        pd.DataFrame(cov_list).to_csv('{}correlated_pairs.csv'.format(self.options.stats_dir), na_rep='NA')
+        return pd.DataFrame(cov_list, columns=['Feat1', 'Feat2', 'CoV']), feat_counts
+        # for feat, count in feat_counts.items():
+
+        # for feat, num in feat_counts.items():
+
+        # cov_list.sort(key=lambda x: x[2], reverse=True)
+        # for i, j, score in cov_list:
+
+

# TODO: Delete groups with no members or just one member.
def _set_descriptor_groups_df(self, grouping_dir=PADEL_GROUPS_DIR, cols=GROUP_COL_NAMES, use3d=False):
@@ -348,16 +389,20 @@
    return df_scaled

# Set Sample weight option.
-    def isolate_observations(self, fit_to=None, sample_wts='auto', contamination=0.001, rstate=None, n_jobs=-1):
+    def isolate_observations(self, fit_to=None, n_samples='auto', sample_wts='auto', sparsity='dense', n_est=100, contamination=0.001, rstate=None, n_jobs=-1):
    from sklearn.ensemble import IsolationForest
-        if not fit_to:
-            fit_to = self.X.feat_frame[self.X.nonzero_vars]
+        predicted = None
+        if type(fit_to) is not pd.DataFrame:
+            fit_to = self.X.feat_frame
    if sample_wts == 'auto':
-            sample_wts = None
-        iso = IsolationForest(n_jobs=n_jobs, contamination=contamination, random_state=0)
-        if fit_to:
-            iso = iso.fit(fit_to, sample_weight=sample_wts)
-        return iso
+            sample_wts = self.options.sample_wts[fit_to.index]
+        iso = IsolationForest(n_jobs=self.options.pjobs, max_samples=n_samples, n_estimators=int(n_est), contamination=contamination, random_state=0)
+        if sparsity == 'dense':
+            dense_df = fit_to[self._X.dense].astype(np.float32)
+            iso = iso.fit(dense_df, sample_weight=self.options.sample_wts)
+            #iso = iso.fit(fit_to[self._X.sparse].astype(), sample_weight=self.options.sample_wts)
+            predicted = iso.predict(dense_df)
+        return iso, predicted

@property
def X(self):
@@ -420,15 +465,17 @@
    col_group, selector = feat_dict


-REQUIRED_OPTS = ('tol_discrete', 'tol_sparse', 'sparsity')
+REQUIRED_OPTS = ()


class FeatureFrame:

def __init__(self, df=None, options=None, *args, **kwargs):
-        self._feat_frame = df
-        self._original = self.feat_frame.copy(deep=True)
    self._options = options
+        self._original = df.copy(deep=True)
+        self._feat_frame = None
+        self._nonzero_vars = None
+        self._zero_vars = None
    self._sparse = None
    self._dense = None
    self._discrete = None
@@ -436,6 +483,8 @@
    self._neg = None
    self._nonneg = None
    self._feat_vars = None
+        self._outliers = None
+        self._inliers = None
    self._feat_stat_scorer = None
    self._corr_pairs = None
    self._corr_mat = None
@@ -443,8 +492,16 @@

@property
def feat_frame(self):
-        return self._feat_frame
+        if self.inliers is not None and not self.inliers.empty:
+            return self._feat_frame.loc[self.inliers]
+        else:
+            return self._feat_frame

+    @feat_frame.setter
+    def feat_frame(self, value):
+        self._feat_frame = value
+
+
@property
def original(self):
    return self._original
@@ -458,13 +515,32 @@
    self._options = format_options(value)

def _set_attr(self):
+        self.feat_frame = self.check_duplicate_feats(X=self.original)
    self._set_feat_vars()
+        # self.feat_frame = self.original[self.nonzero_vars]
    self._set_sparse_dense()
    self._set_discrete_cont()
    self._set_neg_nonneg()
-        self._set_corr_mat()
+        # self.calc_cov_mat(sample_wts=self.options.sample_wts[self.feat_frame.index])
    # self.scalers()
-        self._set_feat_stat_scorer(None)
+        # self._set_feat_stat_scorer(None)
+
+    # Make results attributes to maintain history of modifications.
+    def check_duplicate_feats(self, X=None):
+        if X is None:
+            X = self.feat_frame
+        dup_col_names = X.columns[X.columns.duplicated(keep=False)]
+        if len(dup_col_names.tolist()) == 0:
+            return None
+        logger.warning('Duplicated features in FeatureFrame:\n{}'.format(dup_col_names.unique()))
+        dict_list = list()
+        renamer_dict = dict()
+        drop_dict, rename_dict = true_duplicates(X, None, subset_names=dup_col_names.unique())
+        [renamer_dict.update(d) for d in dict_list]
+        [logger.info('Dropped indices {} for feature {}'.format(k, v) for k, v in renamer_dict.items())]
+        new_frame = X.copy()
+        [new_frame.drop(index=a, inplace=True) for a in renamer_dict.values()]
+        return new_frame

@property
def feat_vars(self):
@@ -472,7 +548,9 @@

def _set_feat_vars(self, skipna=True, numeric_only=False, *args, **kwargs):
    feat_vars = self.original.var(axis='index', skipna=skipna, numeric_only=numeric_only)
-        logger.info('Feature variances: {}'.format(feat_vars))
+        self._zero_vars = feat_vars[feat_vars == 0].index
+        self._nonzero_vars = feat_vars[feat_vars != 0].index
+        logger.info('Feature variances:\n{}'.format(feat_vars))
    self._feat_vars = feat_vars
    if feat_vars.empty or feat_vars.dropna().empty:
        raise ValueError
@@ -480,6 +558,14 @@
    # self.nonzero_vars((self.feat_vars > 0).index.append(pd.Index([c for c in self.original.columns if c not in self.feat_vars.index])))
    # self.zero_vars(self.original.columns.symmetric_difference(self.nonzero_vars))

+    @property
+    def outliers(self):
+        return self._outliers
+
+    @property
+    def inliers(self):
+        return self._inliers
+
@property
def desc_groups(self):
    return self._desc_groups
@@ -510,6 +596,12 @@
def discrete(self):
    return self._discrete

+    @feat_frame.setter
+    def feat_frame(self, value):
+        if type(value) is not pd.DataFrame:
+            raise ValueError
+        self._feat_frame = value
+
@discrete.setter
def discrete(self, value):
    if type(value) is not pd.Index:
@@ -527,50 +619,38 @@
    self._cont = value

def _set_discrete_cont(self):
-        nonzeros = self.feat_frame[self.nonzero_vars].copy()
+        nonzeros = self.feat_frame.copy().astype(float)
    remainder = nonzeros.round(0).sub(nonzeros).abs()
-        logger.debug(remainder)
+        logger.debug('Rounding remainders:\n{}'.format(remainder))
    self.discrete = nonzeros.columns[(remainder < self.options.tol_discrete).all()]
    self.cont = nonzeros.columns.symmetric_difference(self.discrete)
+        assert not self.discrete.empty
+        assert not self.cont.empty
+        pd.Index
    # logger.info('Discrete: {}', self.discrete)
    # logger.info('Continuous: {}', self.cont)

@property
-    def corr_mat(self):
-        return self._corr_mat
-
-    # aweights is used to balance covariance estimates for different subsets of observations.
-    def _set_corr_mat(self, corr_method='pearson', filter_nan=False, *args, **kwargs):
-        _corr = self.feat_frame[self.dense].corr(method=corr_method)
-        np.cov()
-        if filter_nan:
-            # Remove features with no non-NaN correlations.
-            na_corr = _corr.isna().astype(int)
-            col_na = _corr[na_corr.sum(axis=0) < _corr.shape[1] - 1]
-            logger.info('Correlation matrix: {}'.format(_corr))
-            logger.info('NA value for cols: {}'.format(col_na))
-            logger.warning('These features have no valid correlations: {}'.format(col_na))
-            corr_matrix = _corr.loc[col_na, col_na]
+    def corr_pairs(self):
+        if self._corr_pairs is None:
+            self._corr_pairs = self.set_corr_pairs(corr_method='kendall')
    else:
-            corr_matrix = _corr
-        logger.debug(corr_matrix)
-        self._corr_mat = corr_matrix
-
-    @property
-    def corr_pairs(self):
-        return self._corr_pairs
+            return self._corr_pairs

-    @corr_pairs.setter
-    def corr_pairs(self, corr_method='kendall', *args, **kwargs):
-        corr_val = self.feat_frame.corr_mat(corr_method=corr_method, *args, **kwargs)
-        logger.info(corr_val, corr_val.shape)
+    def set_corr_pairs(self, corr_method='kendall', *args, **kwargs):
+        if self.corr_mat is None:
+            logger.warning('Covariance matrix has not been calculated.')
+            return None
+        else:
+            corr_val = self.feat_frame.corr_mat.copy(deep=True)
+        logger.info('{}\n{}'.format(corr_val, corr_val.shape))
    corr_sorted = corr_val.unstack().sort_values(kind='quicksort', ascending=False)
    logging.info('Sorted Correlation values:\n{}'.format(corr_sorted[:10]))
    corr_set = set()
    for i in range(0, self.X.dense.shape[1]):
        for j in range(0, i + 1):
            corr_set.add((self.X.dense.columns[i], self.X.dense.columns[j]))
-        self._corr_pairs = corr_set
+        self.set_corr_pairs = corr_set

@property
def sparse(self):
@@ -610,8 +690,8 @@
        freq_cut = self.options.tol_sparse / self.X.vary.shape[0]
    else:
        raise ValueError
-        logger.info('Sparse tolerance: {}'.format(self.options.tol_sparse))
-        logger.info('Freq cut: {}'.format(freq_cut))
+        # logger.info('Sparse tolerance: {}'.format(self.options.tol_sparse))
+        # logger.info('Freq cut: {}'.format(freq_cut))
    sparse_list, dense_list, zero_list, freq_dict = list(), list(), list(), dict()
    for col, ser in self._pd_feat_freq(ignore_nan=ignore_nan):
        if all_freq:
@@ -636,7 +716,7 @@
@cachetools.cached(cache={})
def _pd_feat_freq(self, ignore_nan=False):
    # logger.info('Feature variances: ', self.feat_vars)
-        for col, ser in self.original.items():
+        for col, ser in self.feat_frame.items():
        yield col, ser.value_counts(normalize=True, dropna=ignore_nan)


@@ -665,3 +745,100 @@

# train_feats = X_train.copy()
# train_labels = train_y.copy()
+
+
+def rename_duplicates(X, y=None, indices='all', name='auto', sep='_', *args, **kwargs):
+    rename_dict = dict()
+    if indices == 'all' and name == 'auto':
+        raise ValueError('Must specify either indices or name.')
+    elif type(name) is list or type(name) is pd.Index:
+        if type(name) is pd.Index:
+            name_list = name.unique().tolist()
+        else:
+            name_list = list(set(name))
+        return_dict = dict()
+        if 'indices' != 'all':
+            for i, one_name in name_list:
+                return_dict[one_name] = rename_duplicates(X, y, indices=indices[i], name=one_name, sep=sep, *args, **kwargs)
+        else:
+            for one_name in name:
+                return_dict[one_name] = rename_duplicates(X, y, indices=indices, name=one_name, sep=sep, *args, **kwargs)
+        return return_dict
+    elif name == 'auto' and indices != 'all':
+        names = X.columns[indices].unique().tolist()
+        assert len(names) == 1
+        name = names[0]
+    if type(name) is str:
+        matches = X.index.get_indexer_for(target=[name])
+        use_ind = matches
+        if indices != 'all':
+            use_ind = [c for c in matches if c in indices]
+        j = 0
+        for i, ind in enumerate(use_ind):
+            assert X.columns[ind] == name
+            while '{}{}{}'.format(name, sep, j) in X.columns.tolist():
+                j += 1
+            # X.columns[ind] = '{}{}{}'.format(name, sep, j)
+            rename_dict[ind] = copy.deepcopy('{}{}{}'.format(name, sep, j))
+    else:
+        raise ValueError('Keyword "name" must be either string, iterable, or pd.Index')
+    return rename_dict
+
+
+def recursive_vif(X, y, indices, n_feats=2, **kwargs):
+    from sklearn.preprocessing import RobustScaler
+    ind = indices
+    if type(ind) is pd.Index:
+        ind = indices.tolist()
+    X_scaled = RobustScaler(with_centering=False, unit_variance=True).fit_transform(X)
+    remain = deepcopy(indices)
+    while len(indices.tolist()) > n_feats:
+        vif = [variance_inflation_factor(X_scaled.iloc[:, indices].to_numpy(), i) for i in
+                             range(len(ind))]
+        maxi = np.argmax(np.array(vif))
+        logger.debug('Eliminating feature #{} with a VIF of {}'.format(maxi, vif[maxi]))
+        remain.pop(maxi)
+    if type(indices) is pd.Index:
+        return pd.Index(remain)
+    else:
+        return remain
+
+
+def eliminate_feats(X, y, indices, score_func, param, mode=None, *args, **kwargs):
+    winners, losers = set(), set()
+    input_dict = {'random_state': 0, **kwargs}
+    score = score_func(X=X[indices], y=y, **input_dict)
+    if mode == 'kbest' or 'percent' in mode or mode is None:
+        n_feats = param
+        if 'percent' in mode:
+            n_feats = np.round(param * len(unique_ind))
+        n_feats = round(param)
+        if n_feats <= 0 or mode is None:
+            n_feats = 1
+            winners = indices[np.argsort(score, kind='stable')[-n_feats:]]
+    elif 'thresh' in mode or 'cut' in mode:
+        if np.sign(param) == -1:
+            logger.warning('Negative param of {} entered. Evaluated `score <= |param|`'.format(param))
+            winners = np.less_equal(score, -param)
+        elif np.sign(param) == 1 or np.sign(param) == 0:
+            winners = np.greater_equal(score, param)
+        winners = indices[winners]
+    else:
+        winners = set(indices[np.argmax(score)])
+    losers = indices[~winners]
+    return winners, losers
+
+def true_duplicates(X: pd.DataFrame, y=None, subset_names=None, **kwargs):
+    if subset_names is None:
+        subset_names = X.columns.tolist()
+    df = X.copy(deep=True)
+    dups = [a for a in df.columns[df.columns.duplicated()].unique() if a in subset_names]
+    rename_dict, drop_dict = dict(), dict()
+    for long_name in dups:
+        name_ind = X.index.get_indexer_for(target=[long_name])
+        true_dups = df.columns[df.T.duplicated(keep=False)]
+        true_ind = tuple(X[true_dups].index.get_indexer_for(target=[long_name]).tolist())
+        drop_dict[long_name] = true_ind
+        unique_ind = tuple([x for x in name_ind if x not in true_ind])
+        rename_dict.update(rename_duplicates(X, y, indices=unique_ind, name=long_name))
+    return drop_dict, rename_dict
Index: dmso_model_dev/qsar_readiness.py
IDEA additional info:
Subsystem: com.intellij.openapi.diff.impl.patch.CharsetEP
<+>UTF-8
