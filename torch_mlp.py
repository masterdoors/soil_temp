import torch
from torch import nn
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from bisect import bisect
import math
#from torch.profiler import profile, record_function, ProfilerActivity

#X[ijk]
#sampleXestimatorXfeatures
class MyMaskedLayer(nn.Module):
    def __init__(self, in_features, out_features, channels,dtype=torch.float64):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, channels, out_features,dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features,dtype=dtype))
        torch.nn.init.kaiming_uniform_(self.weight)

        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)        

    def forward(self, x, mask):
        return torch.einsum('ijk,kjn->ijn',x, self.weight)[mask].reshape(x.shape[0],-1,self.weight.shape[2]).mean(axis=1) + self.bias

class MaskedPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, channels):    
      super(MaskedPerceptron, self).__init__()
      self.fc1 = MyMaskedLayer(input_size, hidden_size,channels,dtype=torch.float64)
      self.fc2 = nn.Linear(hidden_size, output_size,dtype=torch.float64)
      self.hidden_activation = nn.ReLU()
      #self.drop = nn.Dropout(p=0.1)

    def forward(self, x, mask = None, bias = None):
        
        h = self.fc1(x, mask)
        h2 = self.hidden_activation(h)

        if bias is not None:
            h3 = h2 + bias    
        else:
            h3 = h2    

        out = self.fc2(h3)
        #out = self.drop(out)
        
        return out, h3

class KVDataset(Dataset):
    def __init__(self, X,y = None,indexes = None, bias = None, device = None):
        assert indexes is not None
        assert device is not None
        assert len(indexes) > 0
        self.device = device        
        self.data = X.to(device=self.device)
        if y is not None:
            self.labels = y.to(device=self.device)
        else:
            self.labels = None
            
        self.indexes = indexes

        mask = torch.zeros(self.data.shape[0],self.data.shape[1],dtype=bool)
        for i,idx in enumerate(self.indexes):
            j = np.ones(len(idx)) * i
            mask[idx,j] = True
        self.mask = mask.to(device=self.device)
        if bias is not None:
            self.bias = bias.to(device=self.device)
        else:
            self.bias = None    

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.labels is not None: 
            if self.bias is not None:            
                return self.data[idx],self.labels[idx],self.mask[idx],self.bias[idx]
            else:
                return self.data[idx],self.labels[idx],self.mask[idx]
        else:
            if self.bias is not None:
                return self.data[idx],self.mask[idx],self.bias[idx]
            else:
                return self.data[idx],self.mask[idx]

class MLPRB:
    def __init__(self,
                alpha=0.0001,
                batch_size=200,
                learning_rate_init=0.001,
                max_iter=200,
                hidden_size = 20,
                tol=1e-4,
                device = "cpu",
                n_estimators=1,
                n_splits=5,
                criterion = nn.MSELoss(),
                verbose=False):
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.device = device
        self.hidden_size = hidden_size
        self.criterion = criterion 
        self.n_estimators = n_estimators
        self.n_splits = n_splits
        
    #@profile
    def fit(self,X,y, indexes = None, test_indexes = None, bias = None,sample_weight = None):
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        #lengths = [x.shape[1] for x in X]
        X = np.swapaxes(np.asarray(X),0,1)

        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        bias = torch.from_numpy(bias)

        self.model = MaskedPerceptron(X.shape[2],self.hidden_size,y.shape[1],X.shape[1])
        self.model.to(device=self.device)
        self.model.device = self.device      

        best_model = self.model  
        best_loss = 100000000  

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate_init,eps=1e-07,weight_decay=self.alpha)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.1,
                                                         patience=10)

        
        batch_size = min(self.batch_size, int(X.shape[0] / 4))
        if bias is not None:
            train_dataset = KVDataset(X, y, indexes,bias,self.device)
            val_dataset = KVDataset(X, y, test_indexes, bias,self.device)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)#,num_workers=4,persistent_workers=True)
            val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=self.batch_size)#,num_workers=4,persistent_workers=True)
        #early_stopping = EarlyStopping(tolerance=25, min_delta=100)
        last_up = -1

        if self.verbose:
            print("Start...")  

        # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU]
        # sort_by_keyword = "cpu_time_total"        
            
        # with profile(activities=activities, record_shapes=True) as prof:
        #    with record_function("model_inference"): 
        
        for epoch in range(self.max_iter):
            eloss = 0.
            steps = 0

            for tensors in train_loader:
                steps += 1
                X_batch, y_batch, mask_batch, bias_batch = tensors
                X_batch = X_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)
                bias_batch = bias_batch.to(device=self.device)
                mask_batch = mask_batch.to(device=self.device)

                optimizer.zero_grad()                         
                output,_ = self.model(X_batch, mask = mask_batch, bias = bias_batch)   
                loss = self.criterion(output.squeeze(), y_batch.double().squeeze())
                eloss += loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                optimizer.step()

            vloss = 0.
            vsteps = 0                
            with torch.no_grad():
                for tensors in val_loader:   
                    vsteps += 1
                    X_batch, y_batch, mask_batch, bias_batch = tensors
                    X_batch = X_batch.to(device=self.device)
                    y_batch = y_batch.to(device=self.device)
                    bias_batch = bias_batch.to(device=self.device)
                    mask_batch = mask_batch.to(device=self.device)

                    output,_ = self.model(X_batch, mask = mask_batch, bias = bias_batch)   
                    loss = self.criterion(output.squeeze(), y_batch.double().squeeze())
                    vloss += loss.item()                    

                vloss = vloss / vsteps

                if vloss < best_loss:
                    best_model = copy.deepcopy(self.model)    
                    best_loss = vloss 
                    last_up = epoch

            scheduler.step(eloss)
            if self.verbose and epoch % 1 == 0:
                print(
                    eloss / steps, vloss, best_loss
                )
            if epoch - last_up > 30:
                break #early stopping    
        #print(prof.key_averages().table(sort_by=sort_by_keyword,row_limit=500))
        
        if self.verbose:
            print("Stop...")
        self.model = best_model    
        
    def decision_function(self,X, indexes = None, bias = None):
        X = np.asarray(X)
        X = np.swapaxes(X,0,1)
        X = torch.from_numpy(X).to(device=self.device)
        bias = torch.from_numpy(bias).to(device=self.device)
        
        #create mask
        repeats = 0
        do_sort = True
        if indexes is None:
            indexes = []
            repeats = self.n_estimators * self.n_splits
            for _ in range(repeats):
                indexes.append(np.arange(0,X.shape[0]))
            do_sort = False    
        else:
            repeats = int(len(indexes) / self.n_splits)       

        output = None
        hidden = None

        if bias is not None:
            dataset = KVDataset(X,None,indexes,bias,self.device) 

            loader = torch.utils.data.DataLoader(dataset, batch_size=256,shuffle=False)       
        
            with torch.no_grad():
                outputs = []
                hiddens = [] 
                for tensors in loader:
                    X_batch, mask_batch, bias_batch = tensors
                    #mask_batch = mask_batch.to(device=self.device)

                    output, hidden = self.model(X_batch, mask = mask_batch, bias = bias_batch)  
                    outputs.append(output.detach().to(torch.device('cpu')))
                    hiddens.append(hidden.detach().to(torch.device('cpu')))

                output = torch.vstack(outputs)
                hidden = torch.vstack(hiddens)

        return output.numpy(), hidden.numpy()                       
