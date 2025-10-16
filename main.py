import pandas as pd
import os
import logging
from src.data.make_dataset import download_kaggle_dataset
from src.features.build_features import build_features
from src.models.train_model import cluster_model, train_model

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format=log_format, level=logging.INFO)

def main():
    file_path= "data/raw/"
    dataset= 'accepted_2007_to_2017.csv'

    if os.path.exists(os.path.join(file_path, dataset)):
        logging.info('Dataset found, reading..')
        raw_df= pd.read_csv(os.path.join(file_path, dataset), low_memory=False)
    else:
        logging.info('Dataset not found, downloading..')
        download_kaggle_dataset(
            dataset="mlfinancebook/lending-club-loans-data", filepath=file_path
        )
        raw_df= pd.read_csv(os.path.join(file_path, dataset), low_memory=False)

    logging.info('Dataset loaded. Preprocessing...')
    build_features(raw_df)
    logging.info('Preprocessing done')


    logging.info('Loading processed interim dataset')
    train= pd.read_csv("data/interim/train.csv")
    test= pd.read_csv("data/interim/test.csv")
    logging.info('Loading done')


    logging.info('Performing risk bucketing and saving clustered data')
    cluster_model(train, test)
    logging.info('Data Clustered and stored.')

    logging.info('Training model')
    train_model(train_path="data/processed/train.csv", test_path="data/processed/test.csv")
    logging.info('Training model done.')

if __name__ == "__main__":
    main()
