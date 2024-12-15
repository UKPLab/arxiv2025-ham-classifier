import pandas as pd
import wandb
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_wandb_data(dataset, sweep_ids, project, entity="xxx", wandb_folder="data/wandb_data"):
    api = wandb.Api()

    summary_list, config_list = [], [] 
    run_name_list, run_id_list = [], []
    sweep_name_list, sweep_id_list = [], []
    history_list = []

    for sweep_id in tqdm(sweep_ids):
        sweep = api.sweep(f"{entity}/{project}/sweeps/{sweep_id}")

        sweep_name_list += [sweep.name] * len(sweep.runs)
        sweep_id_list += [sweep_id] * len(sweep.runs)

        runs = sweep.runs
        for run in runs:
            # .summary contains output keys/values for
            # metrics such as accuracy.
            #  We call ._json_dict to omit large files
            # summary_list.append(run.summary._json_dict)
            summary_list.append(run.summary._json_dict)

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

            # .name is the human-readable name of the run.
            run_name_list.append(run.name)
            run_id_list.append(run.id)

            # Put history in csv file
            history = run.scan_history()
            history_df = pd.DataFrame(history)
            history_df['run_id'] = run.id
            if 'epoch' not in history_df.columns:
                continue

            history_df = history_df[history_df['epoch'].notna()]
            history_list.append(history_df)

    runs_df = pd.DataFrame(
        {"name": run_name_list, "id": run_id_list,
        "sweep_name": sweep_name_list, "sweep_id": sweep_id_list}
    )
    config_df = pd.DataFrame(config_list)
    # print(summary_list)
    summary_df = pd.DataFrame([sl for sl in summary_list])


    info_df = pd.concat([runs_df, summary_df, config_df], axis=1)
    # Put runs in csv file
    info_df.to_csv(f"{wandb_folder}/{dataset}_run_configs.csv")

    if len(history_list) != 0:
        history_df = pd.concat(history_list)
        history_df.to_csv(f"{wandb_folder}/{dataset}_history.csv")
    print(f"Done fetching {dataset} data from wandb")


def compute_dataset_aggregates(dataset, wandb_folder="data/wandb_data"):
    info_df = pd.read_csv(f"{wandb_folder}/{dataset}_run_configs.csv")

    agg_df = info_df.groupby("sweep_id", as_index=False).agg({'sweep_name' : 'first', 'n_all_params': 'first',
                                    'dev accuracy': ['mean', 'std'], 'dev F1': ['mean', 'std'], 'dev loss': ['mean', 'std'],
                                    'train accuracy': ['mean', 'std'], 'train F1': ['mean', 'std'], 'loss': ['mean', 'std']})
    agg_df.columns = ['sweep_id', 'sweep_name', 'n_all_params', 'dev accuracy_mean', 'dev accuracy_std', 'dev F1_mean', 'dev F1_std', 'dev loss_mean', 'dev loss_std',
                        'train accuracy_mean', 'train accuracy_std', 'train F1_mean', 'train F1_std', 'train loss_mean', 'train loss_std']
    # Formats columns as mean +- std e.g. .739 +- .010 
    agg_df = agg_df.round(3).astype(str)
    agg_df['Train Accuracy'] = agg_df['train accuracy_mean'] + '$\\pm$' + agg_df['train accuracy_std']
    agg_df['Train F1'] = agg_df['train F1_mean'] + '$\\pm$' + agg_df['train F1_std']
    agg_df['Train Loss'] = agg_df['train loss_mean'] + '$\\pm$' + agg_df['train loss_std']
    agg_df['Dev Accuracy'] = agg_df['dev accuracy_mean'] + '$\\pm$' + agg_df['dev accuracy_std']
    agg_df['Dev F1'] = agg_df['dev F1_mean'] + '$\\pm$' + agg_df['dev F1_std']
    agg_df['Dev Loss'] = agg_df['dev loss_mean'] + '$\\pm$' + agg_df['dev loss_std']

    # Drop columns
    agg_df.drop(['sweep_id', 'train loss_mean', 'train loss_std', 'train accuracy_mean', 'train accuracy_std', 'train F1_mean', 
                'train F1_std', 'dev loss_mean', 'dev loss_std', 'dev accuracy_mean', 'dev accuracy_std', 'dev F1_mean', 'dev F1_std'], axis=1, inplace=True)
    # Print to latex
    latex_table = agg_df.to_latex(f"{wandb_folder}/{dataset}_agg_runs.tex", index=False,
                    column_format="lcccccc")
    print(latex_table)

def compute_all_aggregates(datasets, wandb_folder="data/wandb_data"):
    agg_dfs = []
    for dataset in datasets:
        if dataset == "sst2":
            continue
        info_df = pd.read_csv(f"{wandb_folder}/{dataset}_run_configs.csv")

        agg_df = info_df.groupby("sweep_id", as_index=False).agg({'sweep_name' : 'first', 'n_all_params': 'first',
                                        'dev accuracy': ['mean', 'std'], 'dev F1': ['mean', 'std'], 'dev loss': ['mean', 'std'],
                                        'train accuracy': ['mean', 'std'], 'train F1': ['mean', 'std'], 'loss': ['mean', 'std'],
                                        'test accuracy': ['mean', 'std'], 'test F1': ['mean', 'std'], 'test loss': ['mean', 'std']})
        agg_df.columns = ['sweep_id', 'sweep_name', 'n_all_params', 'dev accuracy_mean', 'dev accuracy_std', 'dev F1_mean', 'dev F1_std', 'dev loss_mean', 'dev loss_std',
                            'train accuracy_mean', 'train accuracy_std', 'train F1_mean', 'train F1_std', 'train loss_mean', 'train loss_std',
                            'test accuracy_mean', 'test accuracy_std', 'test F1_mean', 'test F1_std', 'test loss_mean', 'test loss_std']
        agg_df['sweep_name'] = agg_df['sweep_name'].apply(lambda x: '_'.join(x.split('_')[1:]))
        agg_df['dataset'] = dataset
        agg_df = agg_df.round(3).astype(str)
        agg_dfs.append(agg_df)
    agg_df = pd.concat(agg_dfs)
    # Pivot to have datasets as columns
    agg_df = agg_df.pivot(index='sweep_name', columns=['dataset'])

    print(agg_df)    