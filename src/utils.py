import pandas as pd
import json

def save_results_to_csv(data, filepath):
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def save_summary_json(summary_dict, filepath):
    with open(filepath, 'w') as f:
        json.dump(summary_dict, f, indent=4)
