### Train ###

bilstmTrain.py repr trainFile modelFileName devFile

inputs:
repr - a,b,c,d
trainFile - train file
modelFileName - output model file name (suff _model_<repr> will be added)
devFile - dev file (if you have none you can put the trainFile as input than dev=train)

outputs:
acc_<modelFileName>_<repr> - accuracy each 500 sentences
model_<modelFileName>_<repr> - trained model

### Test ###
assume bilstmTrain.py and bilstmTag.py are in the same dir

bilstmTag.py repr modelFile inputFile modelFileDict outputFileName 

inputs
repr - a,b,c,d
modelFile - model file
inputFile - test file
modelFileDict - model file dictionary
outputFileName - output file name

outputs:
outputFileName - tagged test