import os


def buildAggregateFile(fromScratchFlag):
    dataDirectory = os.curdir + '\data\\'
    aggFile = open(os.curdir + "\data\\aggregate_CSV.csv","w")

    aggFile.truncate(0)
    aggFile.write("Ticker,Date,Open,High,Low,Close,Adj Close,Volume")

    for stockfilePath in os.scandir(dataDirectory):
        print(stockfilePath.path)
        if "_CSV.csv" not in str(stockfilePath.path):
            stockfile = open(stockfilePath, "r")
            stockfile.readline() #skip the first header line
            ticker = str(stockfilePath.path).removeprefix(".\data\\")
            ticker = str(ticker).removesuffix(".csv")
            for line in stockfile:
                lineWithTicker = ticker + "," + line
                print(lineWithTicker)
                aggFile.write(lineWithTicker)

    aggFile.close()

def buildCloseDatedOnlyAggregateFile():
    dataDirectory = os.curdir + '\data\\'
    closedDatedFilePath = os.curdir + "\data\\closeDated_CSV.csv"
    closeDatedFile = open(closedDatedFilePath, "a+")

    closeDatedFile.truncate(0)
    headerLine = "timestamp"

    filesProcessedCount = 0

    for stockfilePath in os.scandir(dataDirectory):
        print(stockfilePath.path)

        ticker = str(stockfilePath.path).removeprefix(".\data\\")
        ticker = str(ticker).removesuffix(".csv")

        headerLine = headerLine + "," + ticker

        if "_CSV.csv" not in str(stockfilePath.path):
            stockfile = open(stockfilePath, "r")

            for line in stockfile:
                #print("NEXT-----------------------")
                params = line.split(",")

                tempFile = open(dataDirectory + "temp.csv", "a+")
                tempFile.truncate(0)

                if(filesProcessedCount == 0):
                    tempFile.write(params[0] + "," + params[4] + "\n")
                else:
                    closeDatedFile.seek(0)
                    #closeDatedFile = open(closedDatedFilePath, "a+")
                    for finalline in closeDatedFile:
                        if params[0] != "Date" and params[0] in finalline:
                            #finalline = finalline.rstrip()
                            tempFile.write(finalline + "," + params[4])
            #replace main with temp
            tempFile.close()
            os.replace(dataDirectory + "temp.csv", closedDatedFilePath)

        filesProcessedCount = filesProcessedCount + 1
        print("Finished processing files: " + str(filesProcessedCount))

def buildCloseDatedOnlyAggregateFileAlt():
    dataDirectory = os.curdir + '\data\\'
    closedDatedFilePath = os.curdir + "\data\\closeDated_CSV.csv"

    #so we hold the whole file in memory
    hashedCDFile = {}

    headerLine = "timestamp"
    filesProcessedCount = 0

    for stockfilePath in os.scandir(dataDirectory):
        print(stockfilePath.path)

        if "_CSV.csv" not in str(stockfilePath.path):
            ticker = str(stockfilePath.path).removeprefix(".\data\\")
            ticker = str(ticker).removesuffix(".csv")
            headerLine = headerLine + "," + ticker

            stockfile = open(stockfilePath, "r")
            for stockLine in stockfile:
                stockLineValues = stockLine.split(",")
                if (filesProcessedCount == 0):
                    hashedCDFile[stockLineValues[0]] = stockLineValues[4]
                else:
                    if(stockLineValues[0] in hashedCDFile):
                        hashedCDFile[stockLineValues[0]] = hashedCDFile[stockLineValues[0]] + "," + stockLineValues[4]
            stockfile.close()

        filesProcessedCount = filesProcessedCount + 1

    #now we have written all the required data to the hashmap in memory, so we can output it
    closeDatedFile = open(closedDatedFilePath, "w")
    closeDatedFile.truncate(0)

    hashedCDFile["Date"] = headerLine
    requireNumberOfLineitems = len(headerLine.split(","))
    closeDatedFile.write(headerLine + "\n")

    for keyDate in hashedCDFile.keys():
        if requireNumberOfLineitems == len(hashedCDFile[keyDate].split(",")) + 1:
            closeDatedFile.write(keyDate + "," + hashedCDFile[keyDate] + "\n")

