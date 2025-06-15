import kagglehub
import pandas as pd

class KaggleDatasetLoader():
    def __init__(
        self,
        kaggle_path:str = "krishnacheedella/telecom-iot-crm-dataset"
        ):
        self.kaggle_path = kaggle_path
        
    def load_dataset(
        self,
        file_name:str=None
        ):
        path = kagglehub.dataset_download(self.kaggle_path)
        
        # Load Customer, Device, and Revenue.
        crm_df = pd.read_csv(f'{path}/crm1.csv')
        device_df = pd.read_csv(f'{path}/crm1.csv')
        rev_df = pd.read_csv(f'{path}/rev1.csv')
        
        merged_df = pd.merge(pd.merge(crm_df, device_df, on='msisdn'),rev_df,on='msisdn')
        merged_df = merged_df.dropna()
        
        if file_name:
            merged_df.to_parquet(
                path=f"./analytics_modules/{file_name}",
                engine='fastparquet'
                )
            print(f"********** The data is already saved as {file_name} **********")
        
        return merged_df

if __name__ == "__main__":
    dataloader = KaggleDatasetLoader()
    dataloader.load_dataset(file_name="test.parquet")