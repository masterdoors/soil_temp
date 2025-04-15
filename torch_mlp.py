import torch
from torch import nn

import numpy as np

class MaskedPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):    
      super(MaskedPerceptron, self).__init__()
      self.fc1 = nn.Linear(input_size, hidden_size,dtype=torch.float64)
      self.fc2 = nn.Linear(hidden_size, output_size,dtype=torch.float64)
      self.hidden_activation = nn.ReLU()

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
        
        return out, h3
    
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
                verbose=False):
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.device = device
        self.hidden_size = hidden_size
        self.criterion =  nn.MSELoss()
        self.n_estimators = n_estimators
        self.n_splits = n_splits

    def fit(self,X,y, indexes = None, bias = None,sample_weight = None):
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        lengths = [x.shape[1] for x in X]
        X = torch.from_numpy(np.hstack(X)).to(device=self.device) 
        y = torch.from_numpy(y).to(device=self.device)

        self.model = MaskedPerceptron(X.shape[1],self.hidden_size,y.shape[1])

        #create mask
        offset = 0
        if indexes is not None:
            masks = []
            for i, idxs in enumerate(indexes):
                mask = np.zeros(X.shape)
                mask[idxs, offset: offset + lengths[i]] = 1.    
                offset += lengths[i]    
                masks.append(mask)
            mask = np.vstack(masks)  
            X = X.repeat((len(indexes),1))
            y = y.repeat((len(indexes),1))     
            if bias is not None:
                bias = np.tile(bias,(len(indexes),1))        
        else:
            mask = None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init,eps=1e-07)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.1,
                                                         patience=10,
                                                         verbose=False)

        if mask is not None and bias is not None:
            train_dataset = torch.utils.data.TensorDataset(X, y, torch.from_numpy(mask).to(device=self.device),
                                                           torch.from_numpy(bias).to(device=self.device))
        else:
            train_dataset = torch.utils.data.TensorDataset(X, y)    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)

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

                
            #with torch.no_grad():
            #     output = self.model(X, mask =  torch.from_numpy(mask).to(device=self.device),
            #                          bias = torch.from_numpy(bias).to(device=self.device)) 
            #     lss = self.criterion(output.flatten(), y.double().flatten())
            scheduler.step(eloss)
            if self.verbose and epoch % 100 == 0:
                print(
                    eloss / steps,
                )          
        
        
    def decision_function(self,X, indexes = None, bias = None):
        lengths = [x.shape[1] for x in X]
        X = torch.from_numpy(np.hstack(X)).to(device=self.device) 
        #create mask
        repeats = 0
        if indexes is not None:
            repeats = len(indexes)
            offset = 0
            masks = []
            for i, idxs in enumerate(indexes):
                mask = np.zeros(X.shape)
                mask[idxs, offset: offset + lengths[i]] = 1.    
                offset += lengths[i]    
                masks.append(mask)
            mask = np.vstack(masks)  
            X = X.repeat((repeats,1))
            if bias is not None:
                bias = np.tile(bias,(repeats,1))   
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