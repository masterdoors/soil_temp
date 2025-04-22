import torch
from torch import nn

import numpy as np
from sklearn.model_selection import train_test_split

class MaskedPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):    
      super(MaskedPerceptron, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size,dtype=torch.float64)
      self.fc2 = nn.Linear(hidden_size, output_size,dtype=torch.float64)
      self.hidden_activation = nn.ReLU()
      #self.drop = nn.Dropout(p=0.1)

    def forward(self, x, mask = None, bias = None):
        if mask is not None:
            masked = x*mask
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
    
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True    
    
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

    def fit(self,X,y, indexes = None, bias = None,sample_weight = None):
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        lengths = [x.shape[1] for x in X]
        X = torch.from_numpy(np.hstack(X)).to(device=self.device) 
        y = torch.from_numpy(y).to(device=self.device)

        self.model = MaskedPerceptron(X.shape[1],self.hidden_size,y.shape[1])
        self.model.to(device=self.device)
        self.model.device = self.device        

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
        else:
            mask = None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init,eps=1e-07)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.1,
                                                         patience=10)

        if mask is not None and bias is not None:
            Xtr, Xtst, ytr,ytst,masktr,masktst,biastr,biastst = train_test_split(X,y,torch.from_numpy(mask).to(device=self.device),
                                                                                torch.from_numpy(bias).to(device=self.device),test_size=0.3)            

            train_dataset = torch.utils.data.TensorDataset(Xtr, ytr, masktr, biastr)
            val_dataset = torch.utils.data.TensorDataset(Xtst, ytst, masktst, biastst)
        else:
            Xtr, Xtst, ytr,ytst = train_test_split(X,y)               
            train_dataset = torch.utils.data.TensorDataset(Xtr, ytr)    
            val_dataset = torch.utils.data.TensorDataset(Xtst, ytst)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset)
        early_stopping = EarlyStopping(tolerance=20, min_delta=10)

        for epoch in range(self.max_iter):
            eloss = 0.
            steps = 0
            for tensors in train_loader:
                steps += 1
                if len(tensors) > 2:
                    X_batch, y_batch, mask_batch, bias_batch = tensors
                else:
                    X_batch, y_batch = tensors
                    mask_batch = None
                    bias_batch = None

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
                    else:
                        X_batch, y_batch = tensors
                        mask_batch = None
                        bias_batch = None

                    output,_ = self.model(X_batch, mask = mask_batch, bias = bias_batch)   
                    loss = self.criterion(output.squeeze(), y_batch.double().squeeze())
                    l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
                    loss += self.alpha * l2_norm
                    vloss += loss.item()                    

                vloss = vloss / vsteps
                early_stopping(eloss / steps,vloss) 
                if early_stopping.early_stop:
                    print("Early stop:", eloss / steps, vloss)
                    break
            scheduler.step(eloss)
            if self.verbose and epoch % 100 == 0:
                print(
                    eloss / steps, vloss
                )          
        
        
    def decision_function(self,X, indexes = None, bias = None):
        lengths = [x.shape[1] for x in X]
        X = torch.from_numpy(np.hstack(X)).to(device=self.device) 
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

        with torch.no_grad():
            if mask is not None:
                output, hidden = self.model(X, mask = torch.from_numpy(mask).to(device=self.device),
                                    bias = torch.from_numpy(bias).to(device=self.device))  
                output = output.reshape((repeats,-1) + (output.shape[1],)).mean(axis=0)
                hidden = hidden.reshape((repeats,-1) + (hidden.shape[1],)).mean(axis=0)

        return output.detach().to(torch.device('cpu')).numpy(), hidden.detach().to(torch.device('cpu')).numpy()                       