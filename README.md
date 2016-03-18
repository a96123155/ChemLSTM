# ChemLSTM
Using LSTM based language model with tokenized chemical name to predict the reaction of organic chemicals

The training data is from chemenet database, which requires Mongo. After that's installed, run the mongo server daemon by doing:

$ mongod &

Then, run 
$ ./create_datasets 1 100000

To generate 10K chemical reactions with 50k real reactions and 50k fake reactions. Then a file train_reactions_100000.json will be generated and then can be used for baseline model as well as the LSTM model to train the chemical vector and predict the reaction. 
