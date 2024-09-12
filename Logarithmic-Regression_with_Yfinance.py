import torch
import numpy as np
import yfinance as yf
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

# Define the custom dataset for stock data
class StockDataset(Dataset):
    def __init__(self, symbol, start_date, end_date):
        self.df = yf.Ticker(symbol).history(start=start_date, end=end_date, interval='1d').reset_index()
        self.df['price_y'] = np.log(self.df['Close'])  # Log of stock price
        self.df['x'] = np.arange(len(self.df))  # Index as feature
        self.x = torch.tensor(self.df['x'].values, dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(self.df['price_y'].values, dtype=torch.float32).view(-1, 1)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Define the Logarithmic Regression model
class LogarithmicRegression(torch.nn.Module):
    def __init__(self, n_input_features):
        super(LogarithmicRegression, self).__init__()
        self.linear = torch.nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        return self.linear(x)

# Parameters
symbol = 'AAPL'
start_date = '2023-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
batch_size = 32
learning_rate = 1e-5  # Change learning rate depending on data size
epochs = 100000

# Create dataset and dataloader
stock_dataset = StockDataset(symbol, start_date, end_date)
dataloader = DataLoader(stock_dataset, batch_size=batch_size, shuffle=True)

# Print dataset details
print("Number of samples in the dataset: " + str(len(stock_dataset)))
first_sample_x, first_sample_y = stock_dataset[0]
print("Datatype of the first training sample (x):", first_sample_x.type())
print("Size of the first training sample (x):", first_sample_x.size())
print("Datatype of the first training sample (y):", first_sample_y.type())
print("Size of the first training sample (y):", first_sample_y.size())

# Initialize the model
n_input_features = 1
model = LogarithmicRegression(n_input_features)
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Initialize weights
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        torch.nn.init.constant_(m.bias, 0)
        
model.apply(init_weights)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    # Print loss every 1000 epochs
    if (epoch + 1) % 10000 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
              #{epoch_loss / len(dataloader):.4f}')

# Plot the results
with torch.no_grad():
    df = stock_dataset.df
    x_tensor = torch.tensor(df['x'].values, dtype=torch.float32).view(-1, 1)  # Convert x to tensor
    df['price_pred'] = model(x_tensor).numpy()  # change tensor back to numpy to plot
    
    # Calculate residuals and standard deviation
    residuals = df['price_y'].values - df['price_pred']
    std_dev = np.std(residuals)
    
    # Create shaded areas ±2 std deviations
    df['upper_bound'] = df['price_pred'] + 1 * std_dev
    df['lower_bound'] = df['price_pred'] - 1 * std_dev
    
    df['highest_bound'] = df['price_pred'] + 2 * std_dev
    df['lowest_bound'] = df['price_pred'] - 2 * std_dev
    
    plt.figure(dpi=100)
    plt.plot(df['Date'], df['price_y'], color='blue', linewidth=0.5, label='Log(Stock Price)')
    plt.plot(df['Date'], df['price_pred'], color='red', linewidth=0.5, label='Regression Line')
    plt.fill_between(df['Date'], df['upper_bound'], df['lower_bound'], color='yellow', alpha=0.3, label='±1 Std Dev')
    plt.fill_between(df['Date'], df['highest_bound'], df['upper_bound'], color='green', alpha=0.3, label='±2 Std Dev')
    plt.fill_between(df['Date'], df['lower_bound'], df['lowest_bound'], color='green', alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Log(Price)')
    plt.title(f'{symbol} Logarithmic Regression with ±2 Std Dev')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
