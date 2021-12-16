import time
import argparse
import random
import pandas as pd

def main(files):
    if len(files) == 1:
        print('One prediction cannot be used for ensemble')
        return

    dfs = [pd.read_csv(file) for file in files]
    for df in dfs:
        assert dfs[0].shape == df.shape
    dfs = [df.sort_values(by='guid/image').reset_index(drop=True) for df in dfs]

    ensemble_df = dfs[0].copy()
    for i in range(dfs[0].shape[0]):
        if len(dfs) <= 3:
            preds = [df.iloc[i,1] for df in dfs]
            if len(set(preds)) == 1:
                ensemble_df.iloc[i,1] = preds[0]
            else:
                pred = random.choice(preds)
                ensemble_df.iloc[i,1] = pred

    timeID = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    ensemble_df.to_csv(f'submission/{timeID}-EN.csv', index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=str, default='', nargs='+', help='the prediction files for ensemble')
    opt = parser.parse_args()
    
    main(opt.files)
