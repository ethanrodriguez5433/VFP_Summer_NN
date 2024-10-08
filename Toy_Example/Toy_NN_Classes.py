import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



class NN(nn.Module):
    '''
    5 hidden layer, 1 input, 1 output Neural network
    '''
    def __init__(self, hidden_size=20, learning_rate=0.001, patience=100, factor=0.5, threshold=1e-4):
        '''
        args:
            float learning_rate: how much should NN correct when it guesses wrong
            int patience: how many repeated values (plateaus or flat data) should occur before changing learning rate
            float factor: by what factor should learning rate decrease upon scheduler step
            float threshold: how many place values to consider repeated numbers
            
        '''
        super(NN, self).__init__()
        
        self.hidden1 = nn.Linear(1, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        self.hidden5 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
        self.loss_data = {
            'loss':[],
            'epoch_count':[]
            }

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', 
                                           factor=factor, patience=patience, threshold=threshold)

        
        
    def forward(self, x):
        '''
        args:
            x: single value or tensor to pass 
        returns:
            output of NN
        '''
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.relu(self.hidden4(x))
        x = self.relu(self.hidden5(x))
        x = self.output(x)

        return x

    
    
    
    def train_model(self, x_train, y_train, num_epochs=1500):
        '''
        args:
            tensor x_train: input dataset to train NN
            tensor y_train: output dataset to train NN
            int num_epochs: iterations of training
        returns:
            dict loss_data: Dictionary with loss data to track training progress
        '''
        x_train = x_train.to(torch.float32)
        y_train = y_train.to(torch.float32)
        
        self.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            outputs = self(x_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            self.loss_data['loss'].append(current_loss)
            self.loss_data['epoch_count'].append(epoch)
            self.scheduler.step(current_loss)

            if (epoch + 1) % (num_epochs/10) == 0:
                print(f'sim NN: Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():.6f}')



    
    
    
    def test_model(self, x_test, y_test):
        '''
        args:
            tensor x_test: an input dataset to pass through NN and test
            tensor y_test: an output dataset to pass through NN
        returns:
            numpy array output: Returns the predictions the NN made with x_test
        '''
        x_test = x_test.to(torch.float32)
        y_test = y_test.to(torch.float32)
        
        self.eval()
        with torch.no_grad():
            output = self(x_test)
            loss = self.criterion(output, y_test).item()
            print(f'Test Loss: {loss:.4f}')

    

    def predict(self, x_value):
        '''
        args:
            x_value: can be single value or array, to get run through NN
        returs:
            numpy array output: outputs from NN for each x point supplied
        '''
        self.eval()
        with torch.no_grad():
            output = self(x_value)
        return output.detach().numpy()








class RegularTransferNN(NN):
    def __init__(self, base_model, lambdas=1e-9, hidden_size=20, 
                 learning_rate=0.001, patience=100, factor=0.5, threshold=1e-4):
        
        super(RegularTransferNN, self).__init__(hidden_size, learning_rate, patience, factor, threshold)
        
        #torch.manual_seed(random_state)
        self.lambdas = lambdas
        self.base_model = base_model
        
        # Copying weights from base_model to this model
        self.hidden1.load_state_dict(base_model.hidden1.state_dict())
        self.hidden2.load_state_dict(base_model.hidden2.state_dict())
        self.hidden3.load_state_dict(base_model.hidden3.state_dict())
        self.hidden4.load_state_dict(base_model.hidden4.state_dict())
        self.hidden5.load_state_dict(base_model.hidden5.state_dict())
        self.output.load_state_dict(base_model.output.state_dict())
        

    def train_model(self, x_train, y_train, num_epochs=1200):
        x_train = x_train.to(torch.float32)
        y_train = y_train.to(torch.float32)
        

        self.train()

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            outputs = self(x_train)
            loss = self.criterion(outputs, y_train)
            
            # Regularization
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in self.parameters():
                l2_reg = l2_reg + torch.norm(param)
            loss = loss + self.lambdas * l2_reg

            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            self.loss_data['loss'].append(loss.detach().numpy())
            self.loss_data['epoch_count'].append(epoch)
            self.scheduler.step(current_loss)

            if (epoch + 1) % (num_epochs / 10) == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
                
                
                
                
                
                
                
class Combined_NN(NN):
    def __init__(self, base_model, hidden_size=20, 
                 learning_rate=0.001, patience=100, factor=0.5, threshold=1e-4):
        super(Combined_NN, self).__init__(hidden_size, learning_rate, patience, factor, threshold)
        
        # Copying weights from base_model to this model
        self.hidden1.load_state_dict(base_model.hidden1.state_dict())
        self.hidden2.load_state_dict(base_model.hidden2.state_dict())
        self.hidden3.load_state_dict(base_model.hidden3.state_dict())
        self.hidden4.load_state_dict(base_model.hidden4.state_dict())
        self.hidden5.load_state_dict(base_model.hidden5.state_dict())
        
        #Add more layers
        self.hidden6 = nn.Linear(hidden_size, hidden_size)
        self.hidden7 = nn.Linear(hidden_size, hidden_size)
        self.hidden8 = nn.Linear(hidden_size, hidden_size)
        self.hidden9 = nn.Linear(hidden_size, hidden_size)
        self.hidden10 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        
        
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.relu(self.hidden4(x))
        x = self.relu(self.hidden5(x))
        x = self.relu(self.hidden6(x))
        x = self.relu(self.hidden7(x))
        x = self.relu(self.hidden8(x))
        x = self.relu(self.hidden9(x))
        x = self.relu(self.hidden10(x))
        x = self.output(x)
        
        return x
    
    def train_model(self, x_train, y_train, num_epochs=10000):
        #for freezing specific layers weights and biases 
        for param in self.hidden1.parameters():
            param.requires_grad = False
        for param in self.hidden2.parameters():
            param.requires_grad = False
        for param in self.hidden3.parameters():
            param.requires_grad = False
        '''
        for param in self.hidden4.parameters():
            param.requires_grad = False
        for param in self.hidden5.parameters():
            param.requires_grad = False
            '''
         
  
        x_train = x_train.to(torch.float32)
        y_train = y_train.to(torch.float32)
        
        self.train()
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            outputs = self(x_train)
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()

            current_loss = loss.item()
            self.loss_data['loss'].append(current_loss)
            self.loss_data['epoch_count'].append(epoch)
            self.scheduler.step(current_loss)

            if (epoch + 1) % (num_epochs/10) == 0:
                print(f'Comb NN: Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():.6f}')
            
        