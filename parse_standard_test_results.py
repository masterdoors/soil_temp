#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('./run_exps.sh')


# In[ ]:


get_ipython().system('pip3.10 install scikit-learn==1.6.0')


# In[ ]:


get_ipython().system('pip3.10 install deep-forest')


# In[ ]:


with open("test_outputs/classic_datasets/boosting_output.txt",'r') as f:
    txt = f.read()
    
    


# In[ ]:


txt = txt.replace("Boosted Forest","Boosted_Forest").replace("Cascade Forest","Cascade_Forest").replace("California housing","California_housing").replace("Liver disorders","Liver_disorders")


# In[ ]:


txt


# In[ ]:


with open("test_outputs/classic_datasets/boosting_output.txt",'w') as f:
    f.write(txt)


# In[ ]:


#parse the log

import pandas as pd

score_ds = pd.read_csv("test_outputs/classic_datasets/boosting_output.txt",sep=" ",header=None)


# In[ ]:


#make average and std values
#model_name,ds_name,depth,max_depth,layers,C,hs,n_trees,n_est,mse_score, mae_score, Y_test.min(),Y_test.max()

avg_dict = {}
for i,row in score_ds.iterrows():
    key = (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8])
    if key not in avg_dict:
        avg_dict[key] = [[],[]]    
    avg_dict[key][0].append(row[9])
    avg_dict[key][1].append(row[10])
        
        


# In[ ]:


import numpy as np

for k in avg_dict:
    avg_dict[k][0] = [np.asarray(avg_dict[k][0]).mean(),np.asarray(avg_dict[k][0]).std()]
    avg_dict[k][1] = [np.asarray(avg_dict[k][1]).mean(),np.asarray(avg_dict[k][1]).std()]                      


# In[ ]:


avg_dict


# In[ ]:


#Table with the best scores


# In[ ]:


best_vals = {}

for model in set(score_ds[0]):
    for ds in set(score_ds[1]):
        if (model,ds) not in best_vals:
            best_vals[(model,ds)] = [1e25,0,0,0,None]
        for k in avg_dict.keys():
            if k[0] == model and k[1] == ds:
                if avg_dict[k][0][0] < best_vals[(model,ds)][0]:
                    best_vals[(model,ds)][0] = avg_dict[k][0][0]
                    best_vals[(model,ds)][1] = avg_dict[k][0][1]
                    best_vals[(model,ds)][2] = avg_dict[k][1][0]
                    best_vals[(model,ds)][3] = avg_dict[k][1][1]
                    best_vals[(model,ds)][4] = k[2:]


# In[ ]:


#depth,max_depth,layers,C,hs,n_trees,n_est,mse_score, mae_score, Y_test.min(),Y_test.max()
datas = []
for k in best_vals:
    datas.append([k[0],k[1],best_vals[k][0],best_vals[k][1],best_vals[k][2],best_vals[k][3],best_vals[k][4][1],best_vals[k][4][2],best_vals[k][4][3],best_vals[k][4][4],best_vals[k][4][5],best_vals[k][4][6]])


# In[ ]:


pd.DataFrame(datas,columns=["Model","Dataset","MSE","MSE std","MAE","MAE std","Tree depth","Layers","C","Hidden size","Trees number","Estimator number/layer"])


# In[ ]:


datas = []
C=[10,100,1000,3000]
n_trees=[5,10,100]
hs=[1, 2, 5, 10]
max_depth=[1, 2]
layers=[1, 3, 5, 10]
datasets=["Diabetes", "California_housing", "Liver_disorders", "KDD98"]

for d in datasets:
    for c in C:
        for tr in n_trees:
            for md in max_depth:
                for l in layers: 
                    row = [d,c,tr,md,l]
                    for h in hs:
                        key = ("Boosted_Forest",d,0,md,l,c,h,tr,50)
                        if key not in avg_dict:
                            res = 100
                        else:
                            res = float(avg_dict[key][0][0]) / best_vals[("Boosted_Forest",d)][0]
                            
                        row.append(res)
                    datas.append(row)    


# In[ ]:


pd.DataFrame(datas,columns=["Dataset","C","Trees number","Tree depth","Layers","HS:1","HS:2","HS:5","HS:10"]).to_csv("boosted_detail_res.csv",sep=";")


# In[ ]:




