from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    obj = DataIngestion()

    train_df, w2_df, w3_df = obj.initiate_data_ingestion()

    print(train_df.shape)
    print(w2_df.shape)
    print(w3_df.shape)