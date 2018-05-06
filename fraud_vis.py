import numpy as np;
import seaborn as sns;


#data_all = pd.read_csv('/Users/lizy/my_doc/Q4/cyber_data_analysis/CS4035 Cyber Data Analytics (201718 Q4) - 4282018 - 441 PM/fraud.csv')

#fraud = data_all.loc[data_all['label']==1] #fraud data



np.random.seed(0)
sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data)