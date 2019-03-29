import Orange
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
scmamp = importr('scmamp')


def compare_no_control(df: pd.DataFrame, use_rank: bool):
    r_df = pandas2ri.py2ri(df)
    friedman_test = scmamp.multipleComparisonTest(r_df, test='iman')
    friedman_p_value = friedman_test.rx2('p.value')[0]
    ranks, _, posthoc_pvals = scmamp.postHocTest(
        data=r_df, test="friedman", correct="hommel", use_rank=use_rank)
    c_pval = pd.DataFrame(
        np.asarray(posthoc_pvals), columns=df.columns, index=df.columns)
    return friedman_p_value, list(ranks), c_pval


def plot_nemenyi_with_cd(df):
    _, avranks, _ = compare_no_control(df, True)

    cd = Orange.evaluation.compute_CD(avranks, df.shape[0])
    Orange.evaluation.graph_ranks(
        avranks,
        list(df.columns),
        cd=cd,
        width=10,
        textspace=1.5
    )
    plt.show()
