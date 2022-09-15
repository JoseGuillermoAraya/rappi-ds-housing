from housing import StandardScaler, np, sns, plt, commons, data_holder


def scale_target():
    DATA = data_holder.get_data()
    price_scaled = StandardScaler().fit_transform(
        np.array(DATA['price'])[:, np.newaxis])
    sns.displot(price_scaled)
    plt.show()
    sns.boxplot(price_scaled)


def drop_outlier_gt_l(x, y, feature_x, feature_y):
    """Drops outlier greater than x and less than y given from a feature
    Parameters
          ----------
          x: number
              X coordinate
          y: number
              Y coordinate
          feature : str
              The name of the feature
    """
    DATA = data_holder.get_data()
    new_data = DATA.drop(DATA[(DATA[feature_x] > x) &
                              (DATA[feature_y] < y)].index)
    data_holder.set_data(new_data)
