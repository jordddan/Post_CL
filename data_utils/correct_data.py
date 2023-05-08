import pandas as pd

data = pd.read_csv("/opt/data/private/qd/noise_master/TC_dataset/imdb/train_aug_en_fr.csv", header=None)
value1 = data.values.tolist()

import pdb
pdb.set_trace()
# num = len(value1)
# data = data.iloc[:,0:2]

# data.to_csv("/opt/data/private/qd/noise_master/TC_dataset/sst2/train.csv",header=None,index=None)

