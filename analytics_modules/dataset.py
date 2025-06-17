import kagglehub
import pandas as pd
import os

class KaggleDatasetLoader():
    def __init__(
        self,
        kaggle_path:str = "krishnacheedella/telecom-iot-crm-dataset"
        ):
        self.kaggle_path = kaggle_path
        
    def load_dataset(
        self,
        ):
        print("Dataset downloading...")
        path = kagglehub.dataset_download(self.kaggle_path)
        data_dict = dict({})
        
        # Load dataset
        for df_name in os.listdir(path):
            file_extention = df_name.split('.')[-1]
            data_path = path + os.sep + df_name
            
            if file_extention == 'csv':
                data_dict[df_name] = pd.read_csv(data_path)
            if file_extention == 'xlsx':
                data_dict[df_name] = pd.read_excel(data_path)
        
        return data_dict

if __name__ == "__main__":
    dataloader = KaggleDatasetLoader(kaggle_path="youssefaboelwafa/clustering-penguins-species")
    data_dict = dataloader.load_dataset()
    raw_data = data_dict['penguins.csv'].dropna()