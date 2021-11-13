import numpy as np
import torch
import matplotlib.pyplot as plt

def myProject(stockDataPath):
    stockDataPath = '..\data\AMZN.csv'
    data = np.genfromtxt(stockDataPath, delimiter=",",usecols=np.arange(0,7), skip_header=True)
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

    input_dim = 1
    hidden_dim = 32
    num_layers = 3
    lstmModel = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    #lstmModel.hidden = lstmModel.init_hidden()


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


myProject('..\data\AMZN.csv')