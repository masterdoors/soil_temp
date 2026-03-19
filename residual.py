
import torch
from torch import nn
import numpy as np
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

class ResBlock(nn.Module):
    def __init__(self,feature_size):
        self.ff = nn.Linear(feature_size, feature_size,dtype=torch.float64,bias=True)
        self.ln = nn.LayerNorm(feature_size)

    def forward(self, x):
        norm_x = self.ln(x)
        y = F.relu(self.ff(norm_x))
        return x + y
         

class ResNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers):    
        super(ResNetModel, self).__init__()
        self.input = nn.Linear(input_size, hidden_size,dtype=torch.float64,bias=True)
        self.blocks = nn.ModuleList([ResBlock(hidden_size) for _ in range(layers)]) 
        self.output = nn.Linear(hidden_size, output_size,dtype=torch.float64,bias=True)

    def forward(self,x):
        embed = self.input(x)
        for b in self.blocks:
            embed = b(embed)
        return self.output(b)        

class ResNetCls:
    def __init__(self,learning_rate_init,layers):
        self.device = 'cpu'
        self.learning_rate_init = learning_rate_init
        self.max_iter = 1000
        self.layers = layers

    def fit(self,X,y):
        num_classes = len(np.unique(y))
        if num_classes < 3:
            self.criterion = BCEWithLogitsLoss
            self.out_size = 1
        else:
            self.criterion = CrossEntropyLoss
            self.out_size = num_classes

        self.model = ResNetModel(X.shape[1],self.hidden_size,self.out_size,self.layers)
        self.model.to(device=self.device)
        self.model.device = self.device      

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate_init,eps=1e-07)

        batch_size = min(self.batch_size, int(X.shape[0] / 4))

        train_dataset = TensorDataset(X, y)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

        old_train_loss = 1e18

        for epoch in range(self.max_iter):
            eloss = 0.
            steps = 0

            for tensors in train_loader:
                steps += 1
                X_batch, y_batch = tensors
                X_batch = X_batch.to(device=self.device)
                y_batch = y_batch.to(device=self.device)

                optimizer.zero_grad()                         
                output,_ = self.model(X_batch)   
                loss = self.criterion(output.squeeze(), y_batch.squeeze())
                eloss += loss.item()

                loss.backward()
                optimizer.step()    

            train_loss = eloss / steps    

            if train_loss > 1.01 * old_train_loss:
                break

            old_train_loss = train_loss  

    def predict(self,X):
        with torch.no_grad:
            X = torch.from_numpy(X)
            res = self.model(X.to(device=self.device))  
            if self.out_size == 1:
                return (res > 0.5).detach().cpu().numpy()
            else:
                return torch.argmax(res,axis=1).detach().cpu().numpy()
