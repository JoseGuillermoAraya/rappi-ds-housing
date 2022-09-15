from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import make_column_transformer, TransformedTargetRegressor
from scipy.special import boxcox1p, inv_boxcox1p
from housing import commons, COLS, models
from housing.commons import data_holder


def sum_two_columns(X):
    result = X.iloc[:, 0] + X.iloc[:, 1]
    return result.to_frame()


def get_feat_names_total_size(self, input_features):
    return ['total_size']


def get_feat_names_group(self, input_features):
    return ['group']


def get_feat_names_has_been_renovated(self, input_features):
    return ['has_been_renovated']


def build_total_size_transformer():
    ft = FunctionTransformer(
        sum_two_columns, feature_names_out=get_feat_names_total_size)
    return make_column_transformer((ft, ['size_house', 'size_basement']))


def build_group_feat_transformer():
    ft = FunctionTransformer(
        lambda X: X.iloc[:, 0].astype(str).str[1:3].to_frame(), feature_names_out=get_feat_names_group)
    return make_column_transformer((ft, ['zip']))


def build_box_cox_transformer():
    ft = FunctionTransformer(lambda x: boxcox1p(
        x, 0), feature_names_out='one-to-one')
    return make_column_transformer((ft, ['size_house', 'avg_size_neighbor_houses', 'size_lot', 'avg_size_neighbor_lot', 'size_basement']))


def build_one_hot_transformer():
    return make_column_transformer((OneHotEncoder(sparse=False), ['is_waterfront', 'zip']))


def build_has_been_renovated_transformer():
    ft = FunctionTransformer(lambda X: X.apply(lambda y: y == 0).astype(
        int), feature_names_out=get_feat_names_has_been_renovated)
    return make_column_transformer((ft, ['renovation_date']))


def build_categories(feature, step):
    [min_feature, max_feature] = commons.get_limits(feature)
    return np.arange(0, max_feature+1, step)


def build_final_pipeline(total_size_transformer, group_feat_transformer, boxcox_transformer, has_been_renovated_transformer, onehot_transformer, vr):
    cols_to_transform = list(
        set(COLS)-set(['num_bed', 'condition', 'year_built', 'num_bath', 'num_floors']))
    id_transformer_for_renovation_date = make_column_transformer((FunctionTransformer(
        lambda x: x, feature_names_out='one-to-one'), ['renovation_date']))
    aux_transformer = FeatureUnion([('total_size_transformer', total_size_transformer), ('group_feat_transformer',
                                                                                         group_feat_transformer), ('has_been_renovated_transformer', has_been_renovated_transformer), ('boxcox_transformer', boxcox_transformer), ('onehot_transformer', onehot_transformer), ('id_transformer_for_renovation_date', id_transformer_for_renovation_date)])
    aux2_transformer = make_column_transformer(
        (aux_transformer, cols_to_transform), remainder='passthrough')
    new_feats_ohe_transformer = make_column_transformer(
        (OneHotEncoder(), [1, 2]), remainder='passthrough')
    transformedTargetReg = TransformedTargetRegressor(
        regressor=vr, func=lambda x: boxcox1p(x, 0), inverse_func=lambda x: inv_boxcox1p(x, 0))
    return make_pipeline(aux2_transformer, new_feats_ohe_transformer, StandardScaler(with_mean=False), transformedTargetReg)


def get_final_estimator():
    xgb_m = models.create_xgb()
    en = models.create_elastic_net()
    r = models.create_kernel_ridge()
    l = models.create_lasso()
    sgd = models.create_sdg()
    vr = models.VotingRegressor([('xgb', xgb_m), ('en', en), ('r', r),
                                ('l', l), ('sgd', sgd)], weights=[0.6, 0.1, 0.1, 0.1, 0.1])
    total_size_transformer = build_total_size_transformer()
    group_feat_transformer = build_group_feat_transformer()
    boxcox_transformer = build_box_cox_transformer()
    has_been_renovated_transformer = build_has_been_renovated_transformer()
    onehot_transformer = build_one_hot_transformer()

    p = build_final_pipeline(total_size_transformer, group_feat_transformer,
                             boxcox_transformer, has_been_renovated_transformer, onehot_transformer, vr)
    data = commons.data_holder.get_data()
    X = data.drop('price', axis=1)
    p.fit(X, data['price'])
    return p
