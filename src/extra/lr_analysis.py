import pandas as pd
import statsmodels.api as sm
from src.config import Config


def run_analysis():
    df_path = r"C:\Users\jsalv\PycharmProjects\exact_sarp\src\extra\for_python.csv"
    df = pd.read_csv(df_path, sep=";", decimal=",")
    df["network_type"] = df["network_type"].apply(lambda x: 1 if x == "R" else 0)

    print(df.head())

    not_a_predictor = {
        "name",
        "type",
        "id",
        "formulation",
        "mip_gap_is_zero",
        "ts_vs_gh_gap",
    }
    predictors = set(df.columns) - not_a_predictor
    target = "mip_gap_is_zero"

    # predictors = {'N', 'C', 'T_max', 'K', 'network_type'}

    X = df[list(predictors)]
    y = df[target]

    # Use statsmodels to fit the linear regression
    x = sm.add_constant(X)
    model = sm.Logit(y, x)
    results = model.fit()
    print(results.summary())

    # Remove the variables that have a p-value greater than 0.2 by looking at the summary
    to_remove = {v for v in predictors if results.pvalues[v] > 0.2}
    predictors -= to_remove
    print("New predictors:", predictors)

    X = df[list(predictors)]
    y = df[target]

    # Use statsmodels to fit the linear regression
    x = sm.add_constant(X)
    model = sm.Logit(y, x)
    results = model.fit()
    print(results.summary())


if __name__ == "__main__":
    config = Config()
    run_analysis()
