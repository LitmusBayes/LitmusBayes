


import os

import pandas as pd

from src.litmusbayes.utils import file_util


def init_df(column_names):
    """
    the method derives from the example below:
    pubs_df = pd.DataFrame(columns=['type', "id", 'title', 'year'])
    for pub in pubs:
        pubs_df.loc[pubs_df.shape[0]] = pub.get_attr_list()
    """
    df = pd.DataFrame(columns=column_names)
    # for a in a_list:
    #     df.loc[df.shape[0]] = a.get_df_list()
    return df


def read_csv(df_path, index_col=0):
    assert os.path.exists(df_path)
    df = pd.read_csv(df_path, index_col=index_col)
    return df


def write_csv(df_path, df):
    df.to_csv(df_path)  # , index=False


def write_df_str(save_path, df, index=True):
    file_util.write_str_to_file(save_path, df.to_string(index=index), False)
