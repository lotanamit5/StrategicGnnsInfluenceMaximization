import pandas as pd
import os

def record(exp_name, args, metrics):
    exp_path = f'{exp_name}.csv'
    
    record_data = {}
    record_data.update(vars(args))
    record_data.update(metrics)

    df = pd.DataFrame([record_data])

    if os.path.exists(exp_path):
        df_existing = pd.read_csv(exp_path)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(exp_path, index=False)