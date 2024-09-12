# Logarithmic_Regression_Nasdaq
PyTorch oriented program to create estimations for investment into based of Yahoo Nasdaq's stock values 

To use the code, one must have PyTorch installed in their desired environment.
The Yahoo Nasdaq stock type can be changed in:
Line 33# symbol = 'AAPL'; Yahoo finance provides a list for the top 100 Nasdaq companies.

Depending on the desired investment time, one can change the start date:
Line 344# start_date = '2023-01-01'; for example

The user will need to vary the learning rate of the model depending on the data set size, this will correlate to the start date. Additionally, the epochs can be chosen to allow for the loss to be minimised. 
The code uses torch tensors, so be careful if additions or alterations are made, one must make sure for plots, the tensors are confired back to numpy dtype. And vise versa for torch packages.

The code uses a "Mean Squared Error" technique to calculate the Loss, I believe this is the best option for Logarithmic regression but I haven't explored PyTorches library to its fullest extent. The code starts of by collecting the data from yfinance and processing it for PyTorch training. This includes taking the log values of the data and convireting to dtype=torch.float32. it follows the steps of:
1) Design model
2) make loss and optimiser
3) Training loop
   -forward pass: compute prediction
   -backward pass: compute gradients
   -update weights

I hope you enjoy reading the code as I did making it.
