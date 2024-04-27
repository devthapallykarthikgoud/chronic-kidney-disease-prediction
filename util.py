import seaborn as sns
import matplotlib.pyplot as plt


def plotter(df, feature: str, *, binrange: tuple=None, binwidth: int=None, xticks=None, **kwargs) :
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.histplot(
        df[feature],
        binwidth=binwidth,
        binrange=binrange,
        ax=ax[0],
        **kwargs,
    )
    sns.boxplot(
        df[feature],
        ax=ax[1]
    )
    
    if xticks :
        ax[0].set_xticks(xticks)
        
def cat_plotter(df, feature: str, **kwargs) :
    sns.barplot(
        data = df[feature].value_counts().to_frame().reset_index(),
        x = feature,
        y = "count",
        **kwargs
    )
