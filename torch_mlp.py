import torch
from torch import nn
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from bisect import bisect
#from torch.profiler import profile, record_function, ProfilerActivity

#import cProfile

# def profile(func):
#    """Decorator for run function profile"""
#    def wrapper(*args, **kwargs):
#        profile_filename = func.__name__ + '.prof'
#        profiler = cProfile.Profile()
#        result = profiler.runcall(func, *args, **kwargs)
#        profiler.dump_stats(profile_filename)
#        return result
#    return wrapper

class MaskedPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):    
      super(MaskedPerceptron, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size,dtype=torch.float64)
      self.fc2 = nn.Linear(hidden_size, output_size,dtype=torch.float64)
      self.hidden_activation = nn.ReLU()
      #self.drop = nn.Dropout(p=0.1)

    def forward(self, x, mask = None, bias = None):
        if mask is not None:
            masked = x.mul(mask)
        else: 
            masked = x
        
        h = self.fc1(masked)
        h2 = self.hidden_activation(h)

        if bias is not None:
            h3 = h2 + bias    
        else:
            h3 = h2    

        out = self.fc2(h3)
        #out = self.drop(out)
        
        return out, h3
    
class KVDataset(Dataset):
    def __init__(self, X,y = None,indexes = None, bias = None,lengths= None, device = None):
        assert indexes is not None
        assert lengths is not None
        assert device is not None
        assert len(indexes) > 0
        #self.device = device        
        #self.data = torch.from_numpy(X).to(device=self.device)
        #if y is not None:
        #    self.labels = torch.from_numpy(y).to(device=self.device)
        #else:
        #    self.labels = None
            
        self.indexes = indexes
        #self.bias = torch.from_numpy(bias).to(device=self.device)
        self.lengths = lengths
        self.cum_lengths = []
        cl = 0
        for l in self.lengths:
            self.cum_lengths.append(cl)
            cl += l
            
        total_size = 0
        self.sizes = []
        for i in self.indexes:
            self.sizes.append(total_size)
            total_size += len(i)
        self.total_len = total_size
        #self.mask = torch.zeros((self.data.shape[1]))

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        batch_num = bisect(self.sizes,idx)
        if batch_num >= len(self.sizes) or self.sizes[batch_num] != idx:
            batch_num -= 1    
        batch_offset = idx - self.sizes[batch_num]
        id_ = self.indexes[batch_num][batch_offset]
        return id_,batch_num

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
        lengths = [x.shape[1] for x in X]
        X = np.hstack(X) 

        X = torch.from_numpy(X).to(device=self.device)
        y = torch.from_numpy(y).to(device=self.device)
        bias = torch.from_numpy(bias).to(device=self.device)

        self.model = MaskedPerceptron(X.shape[1],self.hidden_size,y.shape[1])
        self.model.to(device=self.device)
        self.model.device = self.device      

        best_model = self.model  
        best_loss = 100000000  

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init,eps=1e-07)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.1,
                                                         patience=10)

        
        if bias is not None:
            train_dataset = KVDataset(X, y, indexes,bias,lengths,self.device)
            val_dataset = KVDataset(X, y, test_indexes, bias,lengths,self.device)

            ranges = []
            for i in range(len(train_dataset.cum_lengths)):
                ranges.append((train_dataset.cum_lengths[i],train_dataset.cum_lengths[i] + train_dataset.lengths[i]))
                
            mask = torch.zeros((self.batch_size,X.shape[1])).to(self.device)
            def custom_collate(data):
                datas = np.array(list(zip(*data)))
                if len(data) != self.batch_size:
                    mask_ = torch.zeros((len(data),X.shape[1])).to(self.device)
                else:
                    mask_ = mask
                    mask_.zero_()
                
                for i in range(self.n_estimators * self.n_splits):
                    mask_[datas[1] == i,ranges[i][0]:ranges[i][1]] = 1.
                return X[datas[0]],y[datas[0]],mask_,bias[datas[0]]  
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True, collate_fn=custom_collate)#,num_workers=4,persistent_workers=True)
            val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=self.batch_size, collate_fn=custom_collate)#,num_workers=4,persistent_workers=True)
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
                #mask_batch = mask_batch.to(device=self.device)

                optimizer.zero_grad()                         
                output,_ = self.model(X_batch, mask = mask_batch, bias = bias_batch)   
                loss = self.criterion(output.squeeze(), y_batch.double().squeeze())
                l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
                loss += self.alpha * l2_norm
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
                    mask_batch = mask_batch.to(device=self.device)

                    output,_ = self.model(X_batch, mask = mask_batch, bias = bias_batch)   
                    loss = self.criterion(output.squeeze(), y_batch.double().squeeze())
                    l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
                    loss += self.alpha * l2_norm
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
        lengths = [x.shape[1] for x in X]
        X = np.hstack(X)
        X = torch.from_numpy(X).to(device=self.device)
        bias = torch.from_numpy(bias).to(device=self.device)
        
        #create mask
        repeats = 0
        if indexes is None:
            indexes = []
            repeats = self.n_estimators * self.n_splits
            for _ in range(repeats):
                indexes.append(np.arange(0,X.shape[0]))
        else:
            repeats = int(len(indexes) / self.n_splits)       

        output = None
        hidden = None

        if bias is not None:
            dataset = KVDataset(X,None,indexes,bias,lengths,self.device) 

            ranges = []
            for i in range(len(dataset.cum_lengths)):
                #ranges.append(np.arange(dataset.cum_lengths[i],dataset.cum_lengths[i] + dataset.lengths[i],1))             
                ranges.append((dataset.cum_lengths[i],dataset.cum_lengths[i] + dataset.lengths[i]))

            mask = torch.zeros((self.batch_size,X.shape[1])).to(self.device)
            def custom_collate(data):
                datas = np.array(list(zip(*data)))
                if len(data) != self.batch_size:
                    mask_ = torch.zeros((len(data),X.shape[1])).to(self.device)
                else:
                    mask_ = mask
                    mask_.zero_()
                
                for i in range(self.n_estimators * self.n_splits):
                    mask_[datas[1] == i,ranges[i][0]:ranges[i][1]] = 1.
                return X[datas[0]], mask_,bias[datas[0]]          
                
            loader = torch.utils.data.DataLoader(dataset, batch_size=256,shuffle=False,collate_fn=custom_collate)       
        
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
                output = output.reshape((repeats,-1) + (output.shape[1],)).mean(axis=0)
                hidden = hidden.reshape((repeats,-1) + (hidden.shape[1],)).mean(axis=0)
        return output.numpy(), hidden.numpy()                       
