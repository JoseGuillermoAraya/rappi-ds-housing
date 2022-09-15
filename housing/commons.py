from housing import sns, plt, np, data_holder


def plot_feature_distribution(feature):
    data = data_holder.get_data()
    sns.displot(data[feature], kde=True)


def plot_distribution(df, title=None, rotate_x_ticks=False):
    """Plots the distribution of the given dataframe

    Parameters:
      --------
      feature: str
        The name of the feature
    """
    fg = sns.displot(df, kde=True, aspect=3)
    if (title):
        fg.ax.set_title(title)
    if (rotate_x_ticks):
        for axes in fg.axes.flat:
            _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=45)


def count_0(feature):
    """Counts the amount of 0 values in the feature

      Parameters:
        --------
        feature: str
          The name of the feature
      """
    data = data_holder.get_data()
    return (data[feature] == 0).sum()


def scatterplot(feature, feature2):
    """Counts the amount of 0 values in the feature

        Parameters:
          --------
          feature: str
            The name of the first feature
          feature2: str
            The name of the second feature
        """
    data = data_holder.get_data()
    sns.scatterplot(x=data[feature], y=data[feature2])


def show_distinct(feature):
    """Shows distinct values of feature

          Parameters:
            --------
            feature: str
              The name of the feature
    """
    data = data_holder.get_data()
    print(f"Distinct values for {feature}")
    print(data[feature].unique())


def get_limits(feature):
    """gets min and max values of feature

            Parameters:
              --------
              feature: str
                The name of the feature
              ignore_zero: boolean
                If it should ignore 0 as limit
    """
    data = data_holder.get_data()
    df = data[feature]
    return [df.min(), df.max()]
