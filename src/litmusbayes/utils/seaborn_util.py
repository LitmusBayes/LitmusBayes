


import seaborn as sns
from matplotlib import pyplot as plt

rc = {'font.sans-serif': ['Noto Sans CJK JP', 'Times New Roman'], 'axes.unicode_minus': False}
# rc = {'font.sans-serif': 'Microsoft YaHei,微软雅黑,Microsoft YaHei Light,微软雅黑 Light', 'axes.unicode_minus': False}
sns.set(style='darkgrid', rc=rc)


def draw_count_graph(df, figsize, plot_category, xlabel, ylabel, output_file):
    """
    refer to:
    https://seaborn.pydata.org/archive/0.11/generated/seaborn.countplot.html
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar.html#matplotlib.axes.Axes.bar
    """
    plt.figure(figsize=figsize)  # figsize=(14, 10)
    # sns.set(style="darkgrid")

    # saturation 饱和度 saturation=0.1
    ax = sns.countplot(x=plot_category, data=df, color="#5975A4", width=0.5)  # color="blue"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate(p.get_height(), (x.mean(), y), ha='center', va='bottom')
        # (p.get_x()+0.12, p.get_height()+0.15))

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def draw_bar_chart(df, figsize, x, y, xlabel, ylabel, output_file):
    """
    refer to:
    https://seaborn.pydata.org/generated/seaborn.barplot.html
    """
    plt.figure(figsize=figsize)  # figsize=(14, 10)
    # sns.set(style="darkgrid")

    # saturation 饱和度 saturation=0.1
    ax = sns.barplot(x=x, y=y, data=df, color="#5975A4", width=0.5)  # color="blue"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate(int(p.get_height()), (x.mean(), y), ha='center', va='bottom')
        # (p.get_x()+0.12, p.get_height()+0.15))

    plt.tight_layout()
    plt.savefig(output_file + ".pdf")
    plt.savefig(output_file + ".png")
    plt.show()


def draw_single_boxplot(df, figsize, y_category, xlabel, ylabel, output_file):
    # sns.set(style="darkgrid")
    plt.figure(figsize=figsize)
    ax = sns.boxplot(y=y_category,
                     data=df,
                     width=0.5,
                     showmeans=True,
                     meanprops={
                         "marker": "D",
                         "markerfacecolor": "white",
                         "markeredgecolor": "white",
                         "markersize": "1.5"
                     })
    # https://seaborn.pydata.org/examples/horizontal_boxplot.html
    # sns.stripplot(y=y_category, data=df, size=4, color=".3", linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    mean = df[y_category].mean()
    ax.text(0, mean + 0.03, "{:.2%}".format(mean), ha="center", color='white', size='10')

    plt.tight_layout()
    plt.savefig(output_file + ".pdf")
    plt.savefig(output_file + ".png")
    plt.show()


def draw_two_boxplot(df, figsize, x_category, y_category, xlabel, ylabel, means, output_file, showfliers=True):
    plt.figure(figsize=figsize)
    sns.set(style='darkgrid', rc=rc, font_scale=1.25)
    ax = sns.boxplot(x=x_category,
                     y=y_category,
                     data=df,
                     width=0.5,
                     showmeans=True,
                     meanprops={
                         "marker": "D",
                         "markerfacecolor": "white",
                         "markeredgecolor": "white",
                         "markersize": "1.5"
                     },
                     showfliers=showfliers)
    # showfliers=False
    # color="#5975A4",
    # boxprops=dict(color="grey"), orient = 'h' , medianprops={'color':'grey'}, capprops={'linestyle':'-','color':'white'}
    # df[df.variable == 'fpga_states_cnt']['value'].mean()
    for i in range(len(means)):
        ax.text(i + 0.07, means[i], "{:.2f}".format((means[i])), ha="center", color='white', size='10')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(output_file + ".pdf")
    plt.savefig(output_file + ".png")
    plt.show()
