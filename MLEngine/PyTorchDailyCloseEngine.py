from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

def myProject(stockDataPath):
    stockDataPath = '..\data\AMZN.csv'

    stringDateConverter = lambda x: datetime.strptime(x.decode('ascii'), '%Y-%m-%d')
    data = np.genfromtxt(stockDataPath, delimiter=",",usecols=np.arange(0,7), skip_header=True)#, converters = {0: stringDateConverter})
    print("data:" + str(data))

    num_rows = data.shape[0]

    size_of_test_set = int(np.round(0.1*num_rows));
    size_of_training_set = num_rows - size_of_test_set;

    #Input data is the first five colummns
    input_training_data = data[:size_of_training_set, 1:-2]
    print("input_training_data:" + str(input_training_data))

    #The output that we are trying to calculate here is just the close price
    output_training_data = data[:size_of_training_set, -2]
    print("output_training_data:" + str(output_training_data))

    #Test data is same format as above but only the test rows
    input_test_data = data[size_of_training_set:, 1:-2]
    output_test_data = data[size_of_training_set:, -2]



    ###################################################################################################

    input_training_data = input_training_data[..., np.newaxis]
    output_training_data = output_training_data[..., np.newaxis]
    input_test_data = input_test_data[..., np.newaxis]
    output_test_data = output_test_data[..., np.newaxis]

    #input_training_data[:, :, 0] = 0
    #output_training_data[:, :, 0] = 0
    #input_test_data[:, :, 0] = 0
    #output_test_data[:, :, 0] = 0

    number_of_prev_days_to_examine = 30  # choose sequence length
    print('input_training_data.shape = ', input_training_data.shape)
    print('output_training_data.shape = ', output_training_data.shape)
    print('input_test_data.shape = ', input_test_data.shape)
    print('output_test_data.shape = ', output_test_data.shape)

    tensor_input_training_data = torch.from_numpy(input_training_data).type(torch.Tensor)
    tensor_input_test_data = torch.from_numpy(input_test_data).type(torch.Tensor)
    tensor_output_training_data = torch.from_numpy(output_training_data).type(torch.Tensor)
    tensor_output_test_data = torch.from_numpy(output_test_data).type(torch.Tensor)

    ###################################################################################################

    #Creation of the model

    # Build model
    #####################
    input_dimensions = 1
    hidden_dimensions = 32
    number_of_features = 1
    output_dimensions = 1
    number_layers = 1

    # Here we define our model as a class
    class LSTM(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim, number_of_features):
            super(LSTM, self).__init__()
            #Setting hidden dimensions
            self.hidden_dim = hidden_dim

            #Setting Number of hidden layers within model - we only want one layer (only h[0] exists)
            self.num_layers = 1

            #batch_size: the number of training examples utilized in one iteration
            #seq_len: 384
            #input_size / num_features: reflects the number of features

            # batch_first=True causes input/output tensors to be of shape
            # LSTM requires input as (seq_len, batch, input_size) - 3 dimensions
            self.lstm = torch.nn.LSTM(input_dim, hidden_dim, number_of_features, batch_first=True)

            # Readout layer
            self.fc = torch.nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

            # Initialize cell state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

            # We need to detach as we are doing truncated backpropagation through time (BPTT)
            # If we don't, we'll backprop all the way to the start even after going through another batch
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            # Index hidden state of last time step
            # out.size() --> 100, 32, 100
            # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
            out = self.fc(out[:, -1, :])
            # out.size() --> 100, 10
            return out

    lstmModel = LSTM(input_dim=input_dimensions, hidden_dim=hidden_dimensions, output_dim=output_dimensions, num_layers=number_layers, number_of_features=number_of_features)

    loss_fn = torch.nn.MSELoss()

    stochastic_gradient_descent_optimiser = torch.optim.SGD(lstmModel.parameters(), lr=0.05)
    #optimiser = torch.optim.Adam(model.parameters(), lr=0.01)



    ###################################################################################################

    #Training the model

    number_of_epochs = 100
    history_of_loss_fn_calculations = np.zeros(number_of_epochs)
    number_of_unroll_steps = number_of_prev_days_to_examine - 1

    for trainingIteration in range(number_of_epochs):
        # Forward pass
        predicted_output_from_training = lstmModel.forward(tensor_input_training_data)

        print('predicted_output_from_training.shape = ', predicted_output_from_training)
        print('tensor_output_training_data.shape = ', tensor_output_training_data)

        loss = loss_fn(predicted_output_from_training, tensor_output_training_data)
        history_of_loss_fn_calculations[trainingIteration] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        stochastic_gradient_descent_optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        stochastic_gradient_descent_optimiser.step()

    ###################################################################################################

    #Visualiztion of training loss improvement
    plt.plot(history_of_loss_fn_calculations, label="Training loss")
    plt.legend()
    plt.show()

    y_test_pred = lstmModel(tensor_input_test_data)
    print("OUTPUT: " + str(y_test_pred))


myProject('..\data\AMZN.csv')