import torch
import matplotlib.pyplot as plt
import numpy
#from numpy import genfromtxt
from sklearn.model_selection import train_test_split

class LSTM(torch.nn.Module):

    def __init__(self, numOutputTypes, numInputParams, numHiddenDimensions, numLayers):
        super(LSTM, self).__init__()

        self.num_classes = numOutputTypes
        self.num_layers = numInputParams
        self.input_size = numHiddenDimensions
        self.hidden_size = numLayers
        self.seq_length = 3

        self.lstm = torch.nn.LSTM(input_size = numInputParams,
                                 hidden_size = numHiddenDimensions,
                                 num_layers = numLayers,
                                 batch_first=True)

        self.fc = torch.nn.Linear(numHiddenDimensions, numOutputTypes)

    def forward(self, input_x):
        h_0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size).requires_grad_()
        c_0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size).requires_grad_()

        #Propogate forward
        input_x.unsqueeze(0)
        input_x.unsqueeze(1)
        out, h_out = self.lstm(input_x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out

def processDataTickerFile(ticker):
    file_path = "..\\data\\" + str(ticker).upper() + ".csv"
    print("file_path: " + file_path)

    all_data = numpy.genfromtxt(file_path, delimiter=',')
    #print("all_data: " + str(all_data))

    all_data.shape
    train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42, shuffle=True)


    #Process training data
    train_output = torch.tensor(train_data[:,4])
    train_input = numpy.delete(train_data, 5, 1) #Drop adj close
    train_input = numpy.delete(train_input, 4, 1) #Drop close
    train_input = torch.tensor(train_input)


    #Process test data
    test_output = torch.tensor(test_data[:,4])
    test_input = numpy.delete(test_data, 5, 1) #Drop adj close
    test_input = numpy.delete(test_input, 4, 1) #Drop close
    test_input = torch.tensor(test_input)


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