# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:39:01 2023

@author: UPRC
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot


# Define the SMAPE loss function
class SMAPELoss(nn.Module):
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 1e-6
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + epsilon
        diff = torch.abs(y_true - y_pred) / denominator
        return torch.mean(diff)
    
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]
        target = self.data[idx + 1 : idx + self.window_size + 1]  # Include the next 99 points

        return window, target

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, window_size, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()

        self.encoder_embedding = nn.Linear(input_size, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.decoder_embedding = nn.Linear(output_size, d_model)  # Adjust input size for decoder
        self.decoder = nn.Linear(d_model, output_size)
        self.window_size = window_size  # Store window size for decoding

    def forward(self, prev_window, current_window):
        batch_size = prev_window.size(0)

        encoded_prev = self.encoder_embedding(prev_window.to(device))
        encoded_prev = encoded_prev.permute(1, 0, 2)
        encoded_prev = self.encoder(encoded_prev)

        encoded_current = self.decoder_embedding(current_window.to(device))
        encoded_current = encoded_current.permute(1, 0, 2)

        decoded_output = []
        for i in range(encoded_current.size(0)):  # Loop up to the sequence length of encoded_current
            output_i = self.decoder(encoded_current[i])
            decoded_output.append(output_i)

        output = torch.stack(decoded_output, dim=1)  # Stack the output tensors along window_size dimension

        return output.to(device)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

input_size = 8  # Number of input features (p1, p2 and t1)
output_size = 8  # Number of output features (p1, p2 and t1)
window_size = 60# Size of the sliding window
d_model = 8  # Dimension of the model
nhead = 8  # Number of attention heads
num_layers = 16  # Number of transformer layers
num_epochs = 10
batch_size = 64
learning_rate = 0.001

df_list = [
    # lista de poços
			pd.read_csv('\\dados\\well_01_train_1.csv'),
]

scaler = StandardScaler()

# Process each dataframe individually
for df in df_list:
    # Read the dataframe and extract necessary columns
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[(df['p1'] > 0) & (df['p2'] > 0)]
    df = df[['timestamp', 'p1', 'p2',  'p3', 'p4', 
             #'p5', 
             't1', 't2', 't3', 't4'
             ]]
    #df = df.dropna()
    df = df.fillna(0)
    #plot para dados wislive
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['p1']/df['p1'].max(), name='p1', line=dict(color='red', width=4)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['p2']/df['p2'].max(), name='p2', line=dict(color='blue', width=4)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['p3']/df['p3'].max(), name='p3', line=dict(color='green', width=4)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['p4']/df['p4'].max(), name='p4', line=dict(color='orange', width=4)))
 #   fig.add_trace(go.Scatter(x=df['timestamp'], y=df['p5'], name='p5', line=dict(color='yellow', width=4)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['t1']/df['t1'].max(), name='t1', line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['t2']/df['t2'].max(), name='t2', line=dict(color='cyan', width=4)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['t3']/df['t3'].max(), name='t3', line=dict(color='forestgreen', width=4)))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['t4']/df['t4'].max(), name='t4', line=dict(color='silver', width=4)))
    fig.update_xaxes(title_text="time")
    fig.update_yaxes(title_text="variaveis")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black')
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=True,
            showline=True,
            showticklabels=True,
        ),
        autosize=True,
        plot_bgcolor='white'
    )
    plot(fig) 
    
    # Normalize the attribute columns
    df[['p1', 'p2', 'p3', 'p4', 
        't1', 't2', 't3', 't4'
        ]] = scaler.fit_transform(df[['p1', 'p2', 'p3', 'p4', 
                                                            't1', 't2', 't3', 't4'
                                                            ]]) 

    # Convert the attribute columns to float32
    df['p1'] = df['p1'].astype(float)
    df['p2'] = df['p2'].astype(float)
    df['p3'] = df['p3'].astype(float)
    df['p4'] = df['p4'].astype(float)
  #  df['p5'] = df['p5'].astype(float)
    df['t1'] = df['t1'].astype(float)
    df['t2'] = df['t2'].astype(float)
    df['t3'] = df['t3'].astype(float)
    df['t4'] = df['t4'].astype(float)

    # Convert the dataframe to a tensor
    data_tensor = torch.tensor(df[['p1', 'p2', 'p3', 'p4', 
                                   't1','t2', 't3', 't4'
                                   ]].values, dtype=torch.float32) 
    
    # Create an instance of the TimeSeriesDataset
    dataset = TimeSeriesDataset(data_tensor, window_size)
    print(dataset.data)

    # Create a DataLoader for batching and shuffling the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TransformerModel(input_size, output_size, window_size, d_model, nhead, num_layers)

# Split dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define loss function
#criterion = nn.MSELoss()
criterion = SMAPELoss()

# Define optimizer
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model = model.to(device)
criterion = criterion.to(device)

# Training loop
train_losses = []
test_losses = []

#%%
for epoch in range(num_epochs):
    model.train()  # Set model in training mode
    total_loss = 0.0

    for batch_data, target in train_dataloader:
        # Get previous and current windows from the batch_data
        prev_window = batch_data[:, :-1].to(device)
        current_window = target[:, :-1].to(device)  # Truncate the target to match the model output

        # Forward pass
        output = model(prev_window, current_window)  # Predict current window

        # Adjust target for training (exclude the first data point)
        target = target[:, 1:].to(device)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Compute average training loss for the epoch
    average_loss = total_loss / len(train_dataloader)
    train_losses.append(average_loss)
    
    # Validation
    model.eval()  # Set model in evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for batch_data, target in val_dataloader:
            # Get previous and current windows from the batch_data
            prev_window = batch_data[:, :-1].to(device)
            current_window = target[:, :-1].to(device)  # Truncate the target to match the model output
            target = target[:, 1:].to(device)  # Adjust target for validation

            output = model(prev_window, current_window)  # Predict current window
            loss = criterion(output, target)
            val_loss += loss.item()

    # Compute average validation loss for the epoch
    average_val_loss = val_loss / len(val_dataloader)
    test_losses.append(average_val_loss)
    # Print training and validation losses for the epoch
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss:.6f} - Val Loss: {average_val_loss:.6f}")
    
#%% # Plotting the losses
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

# Save the trained model
torch.save(model.state_dict(), '\\transformer_model_art1.pt')
print(model)

 #%%
 # Load the trained model
model = TransformerModel(input_size, output_size, window_size, d_model, nhead, num_layers).to(device)
model.load_state_dict(torch.load('\\transformer_model_art1.pt'))
  
df_list_pred = [
         pd.read_csv('\\dados\\well_A_test_1.csv'),
         pd.read_csv('\\dados\\well_A_test_2.csv'),
         pd.read_csv('\\dados\\well_A_test_3.csv'),
  ]
    
window_size_anomaly = 10*window_size  # You can adjust this window size

for df_pred in df_list_pred:
        
        #df_pred.loc[df_pred['class']==102, 'class']= 2 #substitui transiente por falha
        #df_pred = df_pred.rename(columns={'P-TPT': 'p1', 'P-PDG': 'p2','T-TPT': 't1'})
        df_pred = df_pred.rename(columns={'TPT_PRESS_ANM': 'p1', 'P_PDG_COL': 'p2','TPT_TEMP_ANM': 't1', 
                                          'P_PDG_AN_SUP': 'p3', 'P_PDG_AN_INF' : 'p4', 'T_PDG_COL' : 't2', 'T_PDG_AN_SUP': 't3', 'T_PDG_AN_INF' :'t4'
                                          }) #wislive
    
        # Preprocess the new dataframe
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])
        df_pred = df_pred[['timestamp', 'p1', 'p2', 'p3', 'p4', 
                           't1', 't2', 't3', 't4',
                           #'class'
                           ]]
        df_pred = df_pred.dropna()
        df_pred_aux = df_pred
        df_pred_aux['class1']=0
        df_pred[['p1', 'p2', 'p3', 'p4', 
                 't1', 't2', 't3', 't4'
                 ]] = scaler.transform(df_pred[['p1', 'p2', 'p3', 'p4', 
                                                                      't1', 't2', 't3', 't4'
                                                                      ]])
        df_pred['p1'] = df_pred['p1'].astype(float)
        df_pred['p2'] = df_pred['p2'].astype(float)
        df_pred['p3'] = df_pred['p3'].astype(float)
        df_pred['p4'] = df_pred['p4'].astype(float)
        df_pred['t1'] = df_pred['t1'].astype(float)
        #df_pred['class'] = df_pred['class'].astype(float)
        df_pred['t2'] = df_pred['t2'].astype(float)
        df_pred['t3'] = df_pred['t3'].astype(float)
        df_pred['t4'] = df_pred['t4'].astype(float)
    
        # Convert the dataframe to a tensor
        data_tensor_pred = torch.tensor(df_pred[['p1', 'p2', 'p3', 'p4', 
                                                 't1', 't2', 't3', 't4' 
                                                 ]].values, dtype=torch.float32)
    
        # Create an instance of the TimeSeriesDataset for prediction
        dataset_pred = TimeSeriesDataset(data_tensor_pred, window_size)
        
    
        # Create a DataLoader for batching the prediction data
        dataloader_pred = DataLoader(dataset_pred, batch_size=batch_size, shuffle=False)
    
        # Set the model in evaluation mode
        model.eval()
    
        predictions = []
        with torch.no_grad():
            for batch_data_pred, _ in dataloader_pred:
                prev_window_pred = batch_data_pred[:, :-1].to(device)
                current_window_pred = batch_data_pred[:, 1:].to(device)
    
                output_pred = model(prev_window_pred, current_window_pred)
                output_pred = output_pred[:, -1, :]  # Take the last time step from each window
                predictions.append(output_pred.cpu())
    
        # Concatenate the predictions into a single tensor
        predictions = torch.cat(predictions, dim=0)
    
        # Denormalize the predictions
        predictions = scaler.inverse_transform(predictions)
        
        # Extract the timestamp values
        timestamps = df_pred['timestamp'].values[window_size:]
        
        # Denormalize the true data
        true_data = scaler.inverse_transform(data_tensor_pred[window_size:].numpy())
    
        # Extract the timestamp values
        timestamps = df_pred['timestamp'].values[window_size:]
        
        # Calculate the error difference
        error_diff = abs((predictions - true_data) / true_data)

        # Calculate the mean and standard deviation of the error differences within a moving window
        rolling_mean = pd.DataFrame(error_diff).rolling(window=window_size_anomaly, min_periods=1).mean().values
        rolling_std = pd.DataFrame(error_diff).rolling(window=window_size_anomaly, min_periods=1).std().values

        # Calculate the threshold for anomalies (e.g., two times the standard deviation)
        anomaly_threshold = 3.0 * rolling_std

        # Determine anomalies based on the threshold
        anomaly_flags = error_diff > anomaly_threshold

        # Extract the timestamp values
        timestamps_anomaly = df_pred_aux['timestamp'].values[window_size_anomaly:]

        # Update 'class1' column based on anomalies
       #df_pred_aux['class1'] = anomaly_flags.any(axis=1).astype(int)
        
        # # Calculate the error difference
        # error_diff = abs((predictions - true_data) / true_data)
    
        # # Calculate the standard deviation of the error differences
        # std_error_diff = np.std(error_diff)
    
        # # Calculate the first derivative of error_diff
        # first_derivative = abs(np.gradient(error_diff, axis=0))
    
        # # Calculate the second derivative of error_diff
        # second_derivative = abs(np.gradient(first_derivative, axis=0))
    
        # # Define the threshold for the first derivative
        # derivative_threshold = (2*std_error_diff)
    
        # # Find the indices where the error exceeds the threshold and the first derivative is higher than the threshold
        # anomaly_indices = np.where(np.logical_or.reduce(error_diff > derivative_threshold, axis=1))[0]
        
        # Extract the timestamp values
        #timestamps_anomaly = df_pred_aux['timestamp'].values[anomaly_indices]
        
        # if len(anomaly_flags) == 0:
        #      mudanca = len(df_pred_aux.index)
        # else:
        #      mudanca = anomaly_flags[0]
        
        # size = len(df_pred_aux.index)
        
        # df_pred_aux.loc[(df_pred_aux.index>mudanca), 'class1'] = 1 
        
            
        # Plot the results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=true_data[:, 0]
                                 /true_data[:, 0].max()
                                 , name='True p1', line=dict(color='red', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=true_data[:, 1]
                                 /true_data[:, 1].max()
                                 , name='True p2', line=dict(color='blue', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=true_data[:, 2]/true_data[:, 2].max(), name='True p3', line=dict(color='green', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=true_data[:, 3]/true_data[:, 3].max(), name='True p4', line=dict(color='orange', width=4)))
        #fig.add_trace(go.Scatter(x=timestamps, y=true_data[:, 3]/true_data[:, 3].max(), name='True p4', line=dict(color='yellow', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=true_data[:, 4]
                                 /true_data[:, 4].max()
                                 , name='True t1', line=dict(color='firebrick', width=4))) 
        fig.add_trace(go.Scatter(x=timestamps, y=true_data[:, 5]/true_data[:, 5].max(), name='True t2', line=dict(color='cyan', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=true_data[:, 6]/true_data[:, 6].max(), name='True t3', line=dict(color='forestgreen', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=true_data[:, 7]/true_data[:, 7].max(), name='True t4', line=dict(color='silver', width=4)))  
       # fig.add_trace(go.Scatter(x=timestamps, y=df_pred_aux['class'], name='class', line=dict(color='black', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y= predictions[:, 0]
                                 /predictions[:, 0].max()
                                 , name='Predicted p1', line=dict(color='red', width=4, dash='dot'))) 
        fig.add_trace(go.Scatter(x=timestamps, y= predictions[:, 1]
                                 /predictions[:, 1].max()
                                 , name='Predicted p2', line=dict(color='blue', width=4, dash='dot'))) 
        fig.add_trace(go.Scatter(x=timestamps, y= predictions[:, 2]/predictions[:, 2].max(), name='Predicted p3', line=dict(color='green', width=4, dash='dot'))) 
        fig.add_trace(go.Scatter(x=timestamps, y= predictions[:, 3]/predictions[:, 3].max(), name='Predicted p4', line=dict(color='orange', width=4, dash='dot'))) 
        fig.add_trace(go.Scatter(x=timestamps, y= predictions[:, 4]
                                 /predictions[:, 4].max()
                                 , name='Predicted t1', line=dict(color='firebrick', width=4, dash='dot'))) 
        fig.add_trace(go.Scatter(x=timestamps, y= predictions[:, 5]/predictions[:, 5].max(), name='Predicted t2', line=dict(color='cyan', width=4, dash='dot'))) 
        fig.add_trace(go.Scatter(x=timestamps, y= predictions[:, 6]/predictions[:, 6].max(), name='Predicted t3', line=dict(color='forestgreen', width=4, dash='dot'))) 
        fig.add_trace(go.Scatter(x=timestamps, y= predictions[:, 7]/predictions[:, 7].max(), name='Predicted t4', line=dict(color='silver', width=4, dash='dot'))) 
        #fig.add_trace(go.Scatter(x=timestamps, y=df_pred_aux['class1'], name='class1', line=dict(color='forestgreen', width=4, dash='dot'))) 
        fig.add_trace(go.Scatter(x=timestamps_anomaly,  y=true_data[:, 0]/true_data[:, 0].max(), name='Anomalies', marker=dict(color='mediumvioletred'), mode='markers',  marker_size=10))
        fig.update_xaxes(title_text="time")
        fig.update_yaxes(title_text="variaveis")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black')
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=True,
                showline=True,
                showticklabels=True,
            ),
            autosize=True,
            plot_bgcolor='white'
        )
        plot(fig)  
        
        #%%
        #resultados
        positivo = len(df_pred_aux.loc[(df_pred_aux['class']==1)])
        print(positivo)
        negativo = len(df_pred_aux.loc[(df_pred_aux['class']==0)])
        print(negativo)

        balanceamento = (positivo/(negativo+positivo))*100
        print("Balanceamento(%): " + str(round(balanceamento,2)))

        numero_linhas = size
        print('numero linhas: ' +str(numero_linhas))

        quantidade_dados = size*3
        print('quantidade_pontos: ' +str(quantidade_dados))
        
        nao_classificado = len(df_pred_aux.loc[df_pred_aux['class'].isna()])
        print('quantidade nao classificado: '+str(nao_classificado))

        TP = len(df_pred_aux.loc[(df_pred_aux['class1']==2) & (df_pred_aux['class']==2)]) #true positive
        print('TP: '+str(TP))

        TN = len(df_pred_aux.loc[(df_pred_aux['class1']==0) & (df_pred_aux['class']==0)]) #true negative
        print('TN: '+str(TN))

        FP = len(df_pred_aux.loc[(df_pred_aux['class1']==2) & (df_pred_aux['class']==0)]) #false positive           
        print('FP: '+str(FP))

        FN = len(df_pred_aux.loc[(df_pred_aux['class1']==0) & (df_pred_aux['class']==2)]) #false negativo
        print('FN: '+str(FN))
          
        ACC = (TP + TN) / (size-nao_classificado)
        print("ACC - acurácia: " + str(ACC))

        PR = (TP/(TP+FP))
        print("PR - precisao: " + str(PR)) 

        REC = (TP/(TP+FN))
        print("REC - Recall: " + str(REC)) 

        SP = TN/(TN+FP)  
        print("SP - Especificidade: " + str(SP)) 

        F1 = 2*(PR*REC)/(PR+REC)
        print("F1 score: " + str(F1))

        ACC_balanceada = (REC + SP) / 2
        print("ACC balanceada: " + str(ACC_balanceada))

        taxa_verdadeiro_positivo = TP/(TP+FP) 
        print('taxa verdadeiro positivo: '+str(taxa_verdadeiro_positivo))

        taxa_falso_positivo = 1-SP
        print('taxa falso positivo: '+str(taxa_falso_positivo))   

        AUC = (((1 + taxa_verdadeiro_positivo)*(1-taxa_falso_positivo))/2)+((taxa_verdadeiro_positivo*taxa_falso_positivo)/2)
        print('AUC: '+str(AUC))  
        
        #%%
        # Plot the results
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=timestamps, y=error_diff[:, 0], name='error p1', line=dict(color='red', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=error_diff[:, 1], name='error p2', line=dict(color='blue', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=error_diff[:, 2], name='error p3', line=dict(color='green', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=error_diff[:, 3], name='True p4', line=dict(color='orange', width=4)))
        #fig.add_trace(go.Scatter(x=timestamps, y=error_diff[:, 3], name='error p5', line=dict(color='yellow', width=4)))
        fig.add_trace(go.Scatter(x=timestamps, y=error_diff[:, 2], name='error t1', line=dict(color='firebrick', width=4))) 
        fig.add_trace(go.Scatter(x=timestamps, y= error_diff[:, 5], name='error t2', line=dict(color='cyan', width=4))) 
        fig.add_trace(go.Scatter(x=timestamps, y= error_diff[:, 6], name='error t3', line=dict(color='forestgreen', width=4))) 
        fig.add_trace(go.Scatter(x=timestamps, y= error_diff[:, 7], name='error t4', line=dict(color='silver', width=4))) 
        fig.update_xaxes(title_text="time")
        fig.update_yaxes(title_text="variaveis")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', showline=True, linewidth=1, linecolor='black')
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=True,
                showline=True,
                showticklabels=True,
            ),
            autosize=True,
            plot_bgcolor='white'
        )
        plot(fig)  

