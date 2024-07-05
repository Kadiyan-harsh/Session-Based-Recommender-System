import numpy as np 
import pandas as pd
import datetime as dt
import os

def cleanData(PATH_TO_ORIGINAL_DATA, PATH_TO_PROCESSED_DATA):
    
    if not os.path.exists(PATH_TO_PROCESSED_DATA):
        os.mkdir(PATH_TO_PROCESSED_DATA)
    
    rawDataPath = os.path.join(PATH_TO_ORIGINAL_DATA, 'yoochoose-clicks.dat')
    testPath = os.path.join(PATH_TO_PROCESSED_DATA, 'rsc15Test.csv')
    trainPath = os.path.join(PATH_TO_PROCESSED_DATA, 'rsc15Train.csv')
    validPath = os.path.join(PATH_TO_PROCESSED_DATA, 'rsc15Valid.csv')
    
    data = pd.read_csv(rawDataPath, sep=',', header=None, usecols=[0, 1, 2], dtype={0:np.int32, 1:str, 2:np.int64})
        
    data.columns = ['SessionId', 'TimeStr', 'ItemId']
    data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    del(data['TimeStr'])
    
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]
    
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_test = session_max_times[session_max_times >= tmax - 86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(testPath , sep=',', index=False)
    
    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_valid = session_max_times[session_max_times >= tmax - 86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]
   
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv(trainPath, sep=',', index=False)
   
    print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
    valid.to_csv(validPath, sep=',', index=False)