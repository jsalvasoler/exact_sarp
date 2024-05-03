import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

from src.config import Config


def network_type():
    print(
        '--------------------------\n'
        '--- Network types test ---\n'
        '--------------------------'
    )
    df = pd.read_csv(config.results_filepath, sep=';', decimal=',')
    df.drop_duplicates(subset=['id', 'formulation', 'type'], inplace=True)

    df = df.loc[
        (df['type'] == 'large') & (df['formulation'] == 'scf_cuts_2_start'),
        ['mip_gap', 'id']
    ].sort_values(by='id')
    x = np.array(df[df['id'] <= 48].mip_gap.values)
    y = np.array(df[df['id'] > 48].mip_gap.values)
    print(f'Len x: {len(x)}, Len y: {len(y)}')
    print(f'Mean x: {np.mean(x)}, Mean y: {np.mean(y)}')

    w = wilcoxon(x, y, method='approx')
    print(w)
    if w.pvalue < 0.05:
        print(' -- Reject null hypothesis => different distributions')
    else:
        print(' -- Accept null hypothesis => same distributions')


def scf_improv():
    print(
        '--------------------------\n'
        '--- SCF improvements -----\n'
        '--------------------------'
    )
    df = pd.read_csv(config.results_filepath, sep=';', decimal=',').sort_values(by='id')
    df.drop_duplicates(subset=['id', 'formulation', 'type'], inplace=True)

    df = df.loc[
        (df['type'] == 'large') &
        ((df['formulation'] == 'scf') | (df['formulation'] == 'scf_cuts_2_start')),
        ['id', 'mip_gap', 'formulation']
    ]
    x = np.array(df[df['formulation'] == 'scf'].mip_gap.values)
    y = np.array(df[df['formulation'] == 'scf_cuts_2_start'].mip_gap.values)

    print(f'Len x: {len(x)}, Len y: {len(y)}')
    print(f'Mean x: {np.mean(x)}, Mean y: {np.mean(y)}')

    w = wilcoxon(x, y)
    print(w)
    if w.pvalue < 0.05:
        print(' -- Reject null hypothesis => different distributions')
    else:
        print(' -- Accept null hypothesis => same distributions')


def scf_vs_ts():
    print(
        '--------------------------\n'
        '--- SCF vs TS ------------\n'
        '--------------------------'
    )
    df = pd.read_csv(config.results_filepath, sep=';', decimal=',')
    df.drop_duplicates(subset=['id', 'formulation', 'type'], inplace=True)

    df = df[(df['type'] == 'large')]
    best_bounds = df.groupby('id').best_bound.min().reset_index()
    df = df.merge(best_bounds, on='id', suffixes=('', '_min'))

    df = df.loc[
        (df['formulation'] == 'scf_cuts_2_start'),
        ['id', 'objective', 'Z_TS', 'best_bound_min']
    ].sort_values('id')
    assert len(df) == 96
    df['TS_gap'] = np.abs(df['Z_TS'] - df['best_bound_min']) / df['Z_TS']
    df['gu_gap'] = np.abs(df['objective'] - df['best_bound_min']) / df['objective']

    x = np.array(df['TS_gap'].values)
    y = np.array(df['gu_gap'].values)

    print(f'Len x: {len(x)}, Len y: {len(y)}')
    print(f'Mean x: {np.mean(x)}, Mean y: {np.mean(y)}')

    w = wilcoxon(x, y)
    print(w)
    if w.pvalue < 0.05:
        print(' -- Reject null hypothesis => different distributions')
    else:
        print(' -- Accept null hypothesis => same distributions')


def case():
    print(
        '--------------------------\n'
        '--- (case) SCF vs TS -----\n'
        '--------------------------'
    )
    df = pd.read_csv(config.results_filepath, sep=';', decimal=',')
    df.drop_duplicates(subset=['id', 'formulation', 'type'], inplace=True)
    df = df.loc[
        (df['type'] == 'case') & (df['formulation'] == 'scf_cuts_2_start'),
        ['mip_gap', 'best_bound', 'Z_TS', 'objective']
    ]
    df['TS_gap'] = np.abs(df['Z_TS'] - df['best_bound']) / df['Z_TS']
    df['gu_gap'] = np.abs(df['objective'] - df['best_bound']) / df['objective']

    x = np.array(df['TS_gap'].values)
    y = np.array(df['gu_gap'].values)

    print(f'Len x: {len(x)}, Len y: {len(y)}')
    print(f'Mean x: {np.mean(x)}, Mean y: {np.mean(y)}')

    w = wilcoxon(x, y)
    print(w)
    if w.pvalue < 0.05:
        print(' -- Reject null hypothesis => different distributions')
    else:
        print(' -- Accept null hypothesis => same distributions')


def main(name):
    match name:
        case 'network_type':
            network_type()
        case 'scf_improv':
            scf_improv()
        case 'scf_vs_ts':
            scf_vs_ts()
        case 'case':
            case()


if __name__ == '__main__':
    config = Config()
    config.results_file = 'big_results.csv'

    main('network_type')
    main('scf_improv')
    main('scf_vs_ts')
    main('case')
