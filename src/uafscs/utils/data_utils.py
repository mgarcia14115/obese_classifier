import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


class Dataset:



    def __init__(self,filepath,dropList,label):
        self.data = self.read_data(filepath)
        self.labels = self.data[label]
        self.data = self.data.drop(labels = dropList,axis=1)
        self.data = self.data.drop(labels=[label],axis=1)
        
        self.scalers  = {}
        self.encoders = {}
        self.label_encoder = LabelEncoder()
    
    def read_data(self,filepath):
        df = pd.read_csv(filepath)
        return df
    
    def transform_categorical(self):
        col = 0
        for type in self.data.dtypes:
            if type != np.float64:
                enc = OneHotEncoder(handle_unknown="ignore",sparse_output=False).set_output(transform='pandas')
                df_trans = enc.fit_transform(np.asarray(self.data[self.data.columns[col]]).reshape(-1,1))
                self.encoders[self.data.columns[col]] = enc
                self.data = pd.concat(ignore_index=False,objs=[self.data,df_trans],axis=1).drop(labels = [self.data.columns[col]],axis=1)
            else:
                col+=1
    
    def scale_data(self):
        cols = self.data.shape[1]
        df_scaled_data = pd.DataFrame(columns=self.data.columns)

        for i in range(cols):
            scaler = MinMaxScaler()
            scaler.fit(np.asarray(self.data.iloc[:,i]).reshape(-1,1))
            scaled_feature = scaler.transform(np.asarray(self.data.iloc[:,i]).reshape(-1,1))
            self.scalers[self.data.columns[i]] = scaler
            df_scaled_data[self.data.columns[i]] = pd.Series(scaled_feature.reshape(-1))

        self.data = df_scaled_data

    def encode_labels(self):
        self.labels = self.label_encoder.fit_transform(self.labels)