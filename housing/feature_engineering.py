from housing import sns, norm, plt, stats, np, pd, NUMERICAL_COLS, boxcox1p, commons, OneHotEncoder, make_column_transformer, OrdinalEncoder, scipy, data_holder


def fit_normal(feature):
    """Plots the distribution of the feature and fits a normal distribution to it
    Parameters
        ----------
        feature : str
            The name of the feature
    """
    DATA = data_holder.get_data()
    sns.distplot(DATA[feature], fit=norm)
    fig = plt.figure()
    res = stats.probplot(DATA[feature], plot=plt)
    plt.show()


def boxcox(feature, l):
    """ Box-Cox-transforms the given feature
    Parameters
        ----------
        feature : str
            The name of the feature
        l: number
            Lambda value for box cox
    """
    DATA = data_holder.get_data()
    DATA[feature] = boxcox1p(DATA[feature], l)
    data_holder.set_data(DATA)


def boxcox_mult(features, l):
    """ Box-Cox-transforms the given features
    Parameters
        ----------
        feature : [str]
            The name of the features
        l: number
            Lambda value for box cox
    """
    for feature in features:
        boxcox(feature, l)


def get_skew():
    DATA = data_holder.get_data()
    skewed_feats = DATA[NUMERICAL_COLS].apply(
        lambda x: x.skew()).sort_values(ascending=False)
    print("\nSkew in features: \n")
    skewness = pd.DataFrame({'Skew': skewed_feats})
    return skewness


def add_total_size():
    """Adds a total size feature 'size_house'+'size_basement'
    """
    DATA = data_holder.get_data()
    DATA['total_size'] = DATA['size_house']+DATA['size_basement']
    data_holder.set_data(DATA)


def get_limits(array):
    """Gets lower and upper limit of an array
    ----------
        array : [number]
            Array of numbers
    """
    return [min(array), max(array)]


def view_zip(digits_place, partition=None):
    """Plots the distribution of the first digit of zip

    ----------
        digits_place : [number]
            Places of the digits to plot
        partition: [number]
            Partition the data by these digits
    ---------
    example: view_zip([4,5],[2,3])
    For each group of 2nd and 3rd digit of zip code plot the 4th and 5th digit
    """
    DATA = data_holder.get_data()
    [min_place, max_place] = get_limits(digits_place)
    if (partition):
        [min_place_partition, max_place_partition] = get_limits(partition)
        group_column = DATA['zip'].astype(
            str).str[min_place_partition-1:max_place_partition]
        groups = group_column.unique()
        aux = pd.concat([DATA['zip'], group_column.rename('group')], axis=1)
        for group in groups:
            commons.plot_distribution(
                aux[aux['group'] == group]['zip'].astype(str).str[min_place-1:max_place], title=f"{min_place} and {max_place} digit distribution for '{group}' {min_place_partition} and {max_place_partition} digit", rotate_x_ticks=True)
    else:
        commons.plot_distribution(
            DATA['zip'].astype(str).str[min_place-1:max_place])


def create_group_feature():
    """Creates a new column 'group' in DATA with the 2nd and 3rd digit of zip
    """
    DATA = data_holder.get_data()
    DATA['group'] = DATA['zip'].astype(str).str[1:3]
    data_holder.set_data(DATA)
    return DATA


def one_hot_encode(features):
    """Uses sklearn one hot encoder to encode the features given
        -------
        Parameters:
            features: [string]
                The name of the columns to encode
    """
    ohe = OneHotEncoder()
    result = transform_features(features, ohe)
    return result


def ordinal_encode(features, step):
    """Uses sklearn ordinal encoder to encode the features given
    The list of the labels is created as the integer numbers from
    min value of the feature to the max value
        -------
        Parameters:
            features: [string]
                The name of the columns to encode
            step: number
                The step to generate the labels
            ignore_zero:
                If zero should be ignored as lower limit
    """
    categories = []
    for i, feature in enumerate(features):
        [min_feature, max_feature] = commons.get_limits(
            feature)
        categories.append(np.arange(
            0, max_feature+1, step))
    print(categories)
    oe = OrdinalEncoder(categories=categories)
    return transform_features(features, oe)


def transform_features(features, skl_transformer):
    """Uses a sklearn transformer to transform the given features
    Returns the data frame with the columns transformed
    -------
        Parameters:
            features: [string]
                The name of the columns to encode
            skl_transformer:
                sklearn transformer to use
    """
    DATA = data_holder.get_data()
    transformer = make_column_transformer(
        (skl_transformer, features), remainder='passthrough')
    transformed = transformer.fit_transform(DATA)
    print(transformed.shape)
    print(type(transformed))
    if (type(transformed) == scipy.sparse._csr.csr_matrix):
        result = pd.DataFrame.sparse.from_spmatrix(
            transformed, columns=transformer.get_feature_names_out())
    else:
        result = pd.DataFrame(
            transformed, columns=transformer.get_feature_names_out())
    result.columns = result.columns.str.replace('remainder__', '')
    data_holder.set_data(result)
    return result


def create_has_been_renovated():
    """Creates a new feature that indicates if the house was renovated
    """
    DATA = data_holder.get_data()
    DATA['has_been_renovated'] = DATA['renovation_date'].copy().apply(
        lambda x: x == 0).astype(int)
    data_holder.set_data(DATA)
    return DATA
