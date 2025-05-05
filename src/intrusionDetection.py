from influxdb import InfluxDBClient
import time
import signal
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta


import gc
import sys


torch.set_num_threads(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
seq_length = 10 
hidden_dim = 64
latent_dim = 32
batch_size = 32
num_epochs = 2000
learning_rate = 0.001

counter = 1
client = None

#Influxdb stuff
measurement = 'ue'
fields = ['tx_pkts', 'tx_errors', 'dl_cqi']
n_features = len(fields)

malicious = []

trained = False
use_influx_data = True

n_features = 3  # Adjust based on the number of features (e.g., tx_pkts, tx_error, cqi)

# RNN Autoencoder model
class RNN_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNN_Autoencoder, self).__init__()
        self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, input_dim)
        self.decoder_rnn = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        # Encoder
        _, (h, _) = self.encoder_rnn(x)
        latent = self.hidden_to_latent(h[-1])
        h_decoded = self.latent_to_hidden(latent).unsqueeze(0)
        c_decoded = torch.zeros_like(h_decoded)
        decoder_input = torch.zeros(x.size(0), x.size(1), hidden_dim, device=x.device)
        x_reconstructed, _ = self.decoder_rnn(decoder_input, (h_decoded, c_decoded))

        del latent, h_decoded, c_decoded, decoder_input
        gc.collect()

        return x_reconstructed

# Initialize model
model = RNN_Autoencoder(input_dim=n_features, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

model_path = "autoencoder_random_data.pth" 


if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully", flush=True)
else:
    torch.save(model.state_dict(), model_path)
    print("Initialized model with random weights and saved.", flush=True)

model.eval()
      
def train_model(model, data_tensor, num_epochs=200, batch_size=32, learning_rate=0.01):

    global trained

    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data_batch,) in enumerate(dataloader):
      
            data_batch = data_batch.to(device)

            optimizer.zero_grad()

            reconstructed = model(data_batch)

            loss = criterion(reconstructed, data_batch)
            print('loss:', loss, flush = True)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            if epoch % 10 == 0 and batch_idx == 0:
                with torch.no_grad():
                    input_sample = data_batch[0].cpu().numpy()
                    reconstructed_sample = reconstructed[0].cpu().numpy()


        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}", flush=True)


        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}", flush=True)

        if (epoch + 1) % 100 == 0:
            #torch.save(model.state_dict(), f"autoencoder_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {epoch + 1}")

    trained = True
    print("Training complete.", flush=True)
    return 1

def fetch_data_from_influxdb(seq_length, n_features, measurement, fields, duration='10m'):
    all_tensors = []
    all_ue_ids = []

    # Query to get distinct UE IDs
    query = f'SELECT * FROM "ue" LIMIT 10'
    print("Querying InfluxDB for unique UE IDs...")
    result = client.query(query)
    points = list(result.get_points())

    if not points:
        print("No UE IDs found in InfluxDB.")
        return None

    ue_ids = list({point.get('ue') for point in points if 'ue' in point})
    print(f"Found UE IDs: {ue_ids}")

    for ue_id in ue_ids:
        field_query = ', '.join([f'mean("{f}") as "{f}"' for f in fields])
        query = f'''
        SELECT {field_query}
        FROM "{measurement}"
        WHERE time > now() - {duration} AND "ue" = '{ue_id}'
        GROUP BY time(1s) fill(linear)
        '''
        print(f"Querying InfluxDB for UE: {ue_id}")
        result = client.query(query)
        points = list(result.get_points())

        if not points:
            print(f"No data found for UE {ue_id}")
            continue

        raw_data = np.array([
            [point.get(f, 0.0) for f in fields]
            for point in points
            if all(point.get(f) is not None for f in fields)
        ], dtype=np.float32)
        
        #Normalize The Data
        mean = np.mean(raw_data, axis=0)
        std = np.std(raw_data, axis=0) + 1e-8
        data = (raw_data - mean) / std

        num_sequences = len(data) // seq_length
        if num_sequences == 0:
            print(f"Not enough data to form sequences for UE {ue_id}")
            continue

        data = data[:num_sequences * seq_length].reshape(num_sequences, seq_length, len(fields))
        tensor_data = torch.tensor(data, dtype=torch.float32)

        all_tensors.append(tensor_data)
        all_ue_ids.extend([ue_id] * num_sequences)

    if not all_tensors:
        print("No valid data sequences found.")
        return None

    final_tensor = torch.cat(all_tensors, dim=0)
    return final_tensor, all_ue_ids

# Inference and anomaly detection
def detect_anomalies(model, data_tensor, threshold=0.4):
    model.eval()
    with torch.no_grad():
        data_tensor = data_tensor.to(device)
        reconstructed = model(data_tensor)
        mse = torch.mean((data_tensor - reconstructed) ** 2, dim=(1, 2))
        anomalies = (mse > threshold).nonzero(as_tuple=True)[0]
        sample_index = 0
        if len(data_tensor) > sample_index:
            input_sample = data_tensor[sample_index].cpu().numpy()
            recon_sample = reconstructed[sample_index].cpu().numpy()

        if len(anomalies) > 0:
            print(f"Anomalies detected at indices: {anomalies.tolist()}", flush=True)
            print(f"First anomaly index (anomalies[0]): {anomalies[0].item()}", flush=True)
            return int(anomalies[0])
        else:
            return -1

# Main entry point
def fetchData():
    print("-- FETCHING DATA FROM INFLUXDB OR RANDOM --", flush=True)

    global client
    global counter
    global model
    global trained

    try:
        if client is None:
            client = InfluxDBClient(
                host='ricplt-influxdb.ricplt.svc.cluster.local',
                port=8086
            )
            client.switch_database('Data_Collector')
    except Exception as e:
        print("IntrusionDetection: Error connecting to InfluxDB", flush=True)
        print("Error Message:", e, flush=True)

    try:
        if use_influx_data:
            # Replace this with actual InfluxDB query if available
            print("Using dummy random data for now (Influx fetch not implemented).", flush=True)
            tensor_data, all_ue_ids = fetch_data_from_influxdb(seq_length, n_features, measurement, fields)
            print("\n--- Result from fetch_data_from_influxdb ---")
            #print(tensor_data, flush = True)
            print(f"All UE IDs: {all_ue_ids}", flush=True)

        else:
            print("Aint got no data", flush = True)
            
        if trained == True:
            result = detect_anomalies(model, tensor_data, threshold=0.1)
            
            if result != -1:
                ue_with_anomaly = all_ue_ids[result]
                print(f"Returning UE ID with anomaly: {ue_with_anomaly}", flush=True)
                return int(ue_with_anomaly)

            
        else:
            print("Training Model", flush = True)
            result = train_model(model, tensor_data, num_epochs=500, batch_size=32, learning_rate=0.01)
            print("Training finished", flush = True)
            
            if os.path.exists(model_path):
                print("Path Exists", flush=True)
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("Model loaded successfully 2", flush=True)

            
            if trained == True:
                result = detect_anomalies(model, tensor_data, threshold=0.1)
                print("Anomalies successfully detected", flush = True)
            
                if result != -1:
                    ue_with_anomaly = all_ue_ids[result]
                    print(f"Returning UE ID with anomaly: {ue_with_anomaly}", flush=True)
                    return int(ue_with_anomaly)
                else:
                    print("No anomalies found", flush=True)
                    return -1 
            
            
            return -1
                    
                    


    except Exception as e:
        print("Intrusion Detection: Error during inference", flush=True)
        print("Error Message:", e, flush=True)
        return -1
        
    print("Reached end of fetchData without returning a result", flush=True)
    return -1
   
#result_from_fetch = fetchData()   

def get_result():
    print("Returning result", flush=True)
    result_from_fetch = fetchData()
    print(f"Result: {result_from_fetch}", flush=True)
    return int(result_from_fetch)
