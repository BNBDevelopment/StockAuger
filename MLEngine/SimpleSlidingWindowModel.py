import pandas
import torch
import matplotlib.pyplot as plt

class LSTM(torch.nn.Module):

    def __init__(self, numOutputTypes, numInputParams, numHiddenDimensions, numLayers):
        super(LSTM, self).__init__()

        self.num_classes = numOutputTypes
        self.num_layers = numInputParams
        self.input_size = numHiddenDimensions
        self.hidden_size = numLayers
        self.seq_length = windowSize

        self.lstm = torch.nn.LSTM(input_size = numInputParams,
                                 hidden_size = numHiddenDimensions,
                                 num_layers = numLayers,
                                 batch_first=True)

        self.fc = torch.nn.Linear(numHiddenDimensions, numOutputTypes)

    def foward(self, input_x):
        h_0 = torch.zeros(self.num_layers, input_x.input_xsize(0), self.hidden_dim).requires_grad_()
        c_0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_dim).requires_grad_()

        #Propogate forward
        ula, h_out = self.lstm(input_x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

def processDataTickerFile(ticker):
    file_path = "..\\data\\" + str(ticker).upper() + ".csv"
    print("file_path: " + file_path)

    all_data = pandas.read_csv(file_path)
    all_data.head()
    #print("all_data: " + str(all_data))

    train_data = all_data.sample(frac=0.8)  # random state is a seed value
    test_data = all_data.drop(train_data.index)

    #Process training data
    train_output = train_data['Close']
    train_input = train_data.drop('Close', axis=1)
    train_input = train_input.drop('Adj Close', axis=1)


    #Process test data
    test_output = test_data['Close']
    test_input = test_data.drop('Close', axis=1)
    test_input = test_input.drop('Adj Close', axis=1)

    return train_input,train_output,test_input,test_output

def trainModel(ticker):

    #Get Data
    train_input, train_output, test_input, test_output = processDataTickerFile(ticker)

    #Init Params
    number_of_epochs = 100
    learning_rate = 0.01
    number_of_input_params = 5
    hidden_size = 2
    number_of_layers = 5
    number_of_output_types = 1

    myModel = LSTM(number_of_output_types, number_of_input_params, hidden_size, number_of_layers)

    #Use Mean Squared Error for Loss Function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate)

    for epoch in range(0,number_of_epochs):
        prediction = myModel(train_input)
        optimizer.zero_grad()

        loss = criterion(prediction, train_output)

        #Back propogate error
        loss.backward()

        optimizer.step()

        print("Epoch: " + str(epoch) + "\tLoss: " + str(loss.item()))

trainModel("AAPL")