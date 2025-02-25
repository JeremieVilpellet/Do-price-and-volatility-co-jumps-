#%% Imports

import numpy as np
import pandas as pd

#%% Data Handler

class FinancialDataHandler:
    
    def __init__(self, file_path = None, file_name = None):
        if file_path and file_name:
            self.path = f"{file_path}/{file_name}"
            self.load_data()
        else:
            print("Wrong path or file name")


    def load_data(self):
    
        self.data = pd.read_csv(self.path)
################################# Enlever la ligne en dessous pour utiliser le dataset complet ##########################################
        #self.data = self.data.head(1000) #On fait des tests sur les 1000 premières lignes
        self.process_data()
        print(f"Data successfully loaded from {self.path}")
        
    def process_data(self):
       try:
           self.data['date'] = pd.to_datetime(self.data['date'], format='%m/%d/%Y')
           self.data['time'] = pd.to_datetime(self.data['time'], format='%H:%M:%S.%f')
           '''
           # Concaténation des deux colonnes time et date : Prends environ 5 mins d'execution !
           self.data['Datetime'] = pd.to_datetime(date_column.astype(str) + ' ' + time_column.astype(str), errors='coerce') 
            # Supprimer les colonnes "date" et "time"
           self.data = self.data.drop(['date', 'time'], axis=1)
           '''
       except Exception as e:
           print(f"An error occurred duraing data processing : {e}")
           
    def minutePrice(self):
        try:
            # Trading Day : 9h30-16h00
            mask = (self.data['time'].dt.time >= pd.to_datetime('8:30:00').time()) & (self.data['time'].dt.time <= pd.to_datetime('15:00:00').time())
            filtered_data = self.data[mask]
            filtered_data['datetime'] = pd.to_datetime(filtered_data['date'].astype(str) + ' ' + filtered_data['time'].astype(str))

            # Minute prices are the mean of all prices exchanged during the same minute 
            filtered_data['datetime'] = filtered_data['datetime'].dt.floor('T').dt.tz_localize(None)
            #print(filtered_data.head(20))
            minute_price_data = filtered_data.groupby('datetime')['price'].mean().reset_index()
            # Fractions of dollars are not traded at more than .00
            minute_price_data['price'] = minute_price_data['price'].round(2)
            return minute_price_data

        except Exception as e:
            print(f"An error occurred during minutePrice processing: {e}")


    


