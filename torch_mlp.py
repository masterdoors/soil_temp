import torch
from torch import nn
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class MaskedPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):    
      super(MaskedPerceptron, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size,dtype=torch.float64)
      self.fc2 = nn.Linear(hidden_size, output_size,dtype=torch.float64)
      self.hidden_activation = nn.ReLU()
      #self.drop = nn.Dropout(p=0.1)

    def forward(self, x, mask = None, bias = None):
        if mask is not None:
            masked = x * mask
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
    def __init__(self, X,y = None,indexes = None, bias = None,sample_weight = None,lengths= None):
        assert indexes is not None
        assert lengths is not None
        assert len(indexes) > 0
        self.batch_size = len(indexes[0])
        self.total_len = self.batch_size * len(indexes)
        self.data = X
        self.labels = y
        self.indexes = indexes
        self.bias = bias
        self.sample_weight = sample_weight

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        batch_num = int(idx / self.batch_size)
        batch_offset = idx % self.batch_size
        id_ = self.indexes[batch_num][batch_offset]
        if self.labels is not None:
            return self.data[id_], self.labels[id_],self.bias[id_],self.sample_weight[id_]
        else:
            return self.data[id_], self.bias[id_],self.sample_weight[id_]

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

    def fit(self,X,y, indexes = None, test_indexes = None, bias = None,sample_weight = None):
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        lengths = [x.shape[1] for x in X]
        X = torch.from_numpy(np.hstack(X)) 
        y = torch.from_numpy(y)

        self.model = MaskedPerceptron(X.shape[1],self.hidden_size,y.shape[1])
        self.model.to(device=self.device)
        self.model.device = self.device      

        best_model = self.model  
        best_loss = 100000000  

        #create mask
        offset = 0
        if indexes is not None:
            masks = []
            xs = []
            ys = []
            bs = []
            for i, idxs in enumerate(indexes):
                mask = np.zeros(X.shape)
                mask[idxs, offset: offset + lengths[i]] = 1.    
                offset += lengths[i]    
                masks.append(mask[idxs])
                xs.append(X[idxs])
                ys.append(y[idxs])
                if bias is not None:
                    bs.append(bias[idxs])
            mask = np.vstack(masks)  
            X = torch.vstack(xs)
            y = torch.vstack(ys)     
            if len(bs) > 0:
                bias = np.vstack(bs)       
            if test_indexes is not None:
                xs = []
                ys = []
                bs = []
                masks = []
                offset = 0
                for i, idxs in enumerate(test_indexes):
                    mask_ = np.zeros(X.shape)
                    mask_[idxs, offset: offset + lengths[i]] = 1.    
                    offset += lengths[i]    
                    masks.append(mask[idxs])
                    xs.append(X[idxs])
                    ys.append(y[idxs])
                    if bias is not None:
                        bs.append(bias[idxs])
                test_mask = np.vstack(masks)  
                test_X = torch.vstack(xs)
                test_y = torch.vstack(ys)     
                if len(bs) > 0:
                    test_bias = np.vstack(bs)                     
            else:
                test_mask = None
        else:
            mask = None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init,eps=1e-07)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.1,
                                                         patience=10)

        if mask is not None and bias is not None:
            if test_mask is None:
                Xtr,Xtst,ytr,ytst,masktr,masktst,biastr,biastst=train_test_split(X,y,torch.from_numpy(mask).to(device=self.device),torch.from_numpy(bias).to(device=self.device),test_size=0.3)
            else:
                Xtr = X
                ytr = y
                masktr = torch.from_numpy(mask)
                biastr = torch.from_numpy(bias)
                Xtst = test_X
                ytst = test_y
                masktst = torch.from_numpy(test_mask)
                biastst = torch.from_numpy(test_bias)
                
            train_dataset = torch.utils.data.TensorDataset(Xtr, ytr, masktr, biastr)
            val_dataset = torch.utils.data.TensorDataset(Xtst, ytst, masktst, biastst)
        else:
            Xtr, Xtst, ytr,ytst = train_test_split(X,y)     
            train_dataset = torch.utils.data.TensorDataset(Xtr, ytr)    
            val_dataset = torch.utils.data.TensorDataset(Xtst, ytst)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True,pin_memory=True,num_workers=4,persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=self.batch_size,pin_memory=True,num_workers=4)
        #early_stopping = EarlyStopping(tolerance=25, min_delta=100)
        last_up = -1

        if self.verbose:
            print("Start...")        
        for epoch in range(self.max_iter):
            eloss = 0.
            steps = 0
            for tensors in train_loader:
                steps += 1
                if len(tensors) > 2:
                    X_batch, y_batch, mask_batch, bias_batch = tensors
                    X_batch = X_batch.to(device=self.device)
                    y_batch = y_batch.to(device=self.device)
                    mask_batch = mask_batch.to(device=self.device)
                    bias_batch = bias_batch.to(device=self.device)
                else:
                    X_batch, y_batch = tensors
                    mask_batch = None
                    bias_batch = None
                    X_batch = X_batch.to(device=self.device)
                    y_batch = y_batch.to(device=self.device)                    

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
                    if len(tensors) > 2:
                        X_batch, y_batch, mask_batch, bias_batch = tensors
                        X_batch = X_batch.to(device=self.device)
                        y_batch = y_batch.to(device=self.device)
                        mask_batch = mask_batch.to(device=self.device)
                        bias_batch = bias_batch.to(device=self.device)                        
                    else:
                        X_batch, y_batch = tensors
                        X_batch = X_batch.to(device=self.device)
                        y_batch = y_batch.to(device=self.device)                        
                        mask_batch = None
                        bias_batch = None

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
            if epoch - last_up > 20:
                break #early stopping    
        if self.verbose:
            print("Stop...")
        self.model = best_model    
        del Xtr
        del ytr
        del Xtst
        del ytst

        if masktr is not None:
            del masktr
        if masktst is not None:
            del masktst
        if biastr is not None:
            del biastr
        if biastst is not None:
            del biastst
        
    def decision_function(self,X, indexes = None, bias = None):
        lengths = [x.shape[1] for x in X]
        X = torch.from_numpy(np.hstack(X))
        #create mask
        repeats = 0
        if indexes is not None:
            repeats = self.n_estimators
            offset = 0
            masks = []
            xs = []
            bs = []
            j = 0
            for i  in range(self.n_estimators):
                mask = np.zeros(X.shape)

                for _ in range(self.n_splits):
                    idxs = indexes[j]
                    mask[idxs, offset: offset + lengths[j]] = 1.    
                    offset += lengths[j]    
                    j += 1
                masks.append(mask)
                xs.append(X)
                if bias is not None:
                    bs.append(bias)
  
            mask = np.vstack(masks)  
            X = torch.vstack(xs)
            if len(bs) > 0:
                bias = np.vstack(bs)        
        else:
            masks = []
            offset = 0
            repeats = self.n_estimators * self.n_splits
            for i in range(repeats):
                mask = np.zeros(X.shape)
                mask[:, offset: offset + lengths[i]] = 1.    
                offset += lengths[i]    
                masks.append(mask)
            mask = np.vstack(masks)  
            X = X.repeat((repeats,1))
            if bias is not None:
                bias = np.tile(bias,(repeats,1))               

        output = None
        hidden = None

        if mask is not None and bias is not None:
            mask = torch.from_numpy(mask)
            bias = torch.from_numpy(bias)
            dataset = torch.utils.data.TensorDataset(X,mask,bias) 

            loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,shuffle=False,pin_memory=True)       
        
            with torch.no_grad():
                outputs = []
                hiddens = [] 
                for tensors in loader:
                    X, mask, bias = tensors
                    X = X.to(device=self.device)
                    mask = mask.to(device=self.device)
                    bias = bias.to(device=self.device)

                    output, hidden = self.model(X, mask = mask, bias = bias)  
                    outputs.append(output.detach().to(torch.device('cpu')))
                    hiddens.append(hidden.detach().to(torch.device('cpu')))

                output = torch.vstack(outputs)
                hidden = torch.vstack(hiddens)
                output = output.reshape((repeats,-1) + (output.shape[1],)).mean(axis=0)
                hidden = hidden.reshape((repeats,-1) + (hidden.shape[1],)).mean(axis=0)
        return output.numpy(), hidden.numpy()                       
