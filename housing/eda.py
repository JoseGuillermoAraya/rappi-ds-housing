from housing import pd, np, sns, plt, commons, data_holder
from housing.commons import plot_feature_distribution
important_10_cols = []


def target_distribution():
    """Shows statistical info for target, plots the target distribution along with the probability density function
    Calculates skewness and kurtosis for target
    """
    DATA = data_holder.get_data()
    print('Price describe: ')
    print(DATA['price'].describe())
    plot_feature_distribution('price')
    plt.text(5e6, 700, f"Skewness: {np.round(DATA['price'].skew(),2)}",
             horizontalalignment='left', verticalalignment='bottom', size='medium', color='black')
    plt.text(5e6, 600, f"Kurtosis: {np.round(DATA['price'].kurt(),2)}",
             horizontalalignment='left', verticalalignment='bottom', size='medium', color='black')


def corrmat():
    """Plots the correlation matrix for the features in Data and a 'zoomed in' correlation
    matrix with the 10 features with largest correlation coef with 'price'
    """
    DATA = data_holder.get_data()
    corrmat = DATA.corr()
    sns.heatmap(corrmat, square=True, annot=True,
                fmt='.2f', annot_kws={'size': 7})
    plt.show()
    global important_10_cols
    important_10_cols = corrmat.nlargest(10, 'price')['price'].index
    zoomed_corrmat = np.corrcoef(DATA[important_10_cols].values.T)
    sns.heatmap(zoomed_corrmat, yticklabels=important_10_cols.values,
                xticklabels=important_10_cols.values, annot=True, fmt='.2f')


def pairplot():
    """Plots the pairplot for all the features and then for the 10 features with the largest correlation coef with 'price'
    """
    DATA = data_holder.get_data()
    sns.pairplot(DATA)
    plt.show()
    sns.pairplot(DATA[important_10_cols])
