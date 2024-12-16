import yaml
from sqlalchemy import create_engine
import pandas as pd


class RDSDatabaseConnector:
    

    def __init__(self, rds_table: str, db_credentials: dict) -> None:
        
        self.db_credentials = db_credentials  
        self.rds_table = rds_table 
        self.csv_file_name = self.rds_table + '.csv' 

    def initialise_sqlalchemy_engine(self):
        
        DATABASE_TYPE = 'postgresql'
        DBAPI = 'psycopg2'
        HOST = self.db_credentials['RDS_HOST']
        USER = self.db_credentials['RDS_USER']
        PASSWORD = self.db_credentials['RDS_PASSWORD']
        DATABASE =self.db_credentials['RDS_DATABASE']
        PORT = 5432

        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}")
        return engine
    

    def get_dataframe(self):
        
        engine = self.initialise_sqlalchemy_engine()
        dataframe = pd.read_sql_table(self.rds_table, engine)
        return dataframe


    def save_data_in_csv(self):
        
        dataframe = self.get_dataframe()
        try:
          dataframe.to_csv(self.csv_file_name, encoding='utf-8', index=False)
          print(f"Extracted data has been successfully saved to the application directory with file name: {self.csv_file_name}.\n")
        except RuntimeError:
          print("Something went wrong.\n")


    def load_dataframe(self):
        
        df = pd.read_csv(Path(self.csv_file_name))
        print(f'Size of DataFrame: [{df.shape[0]} rows x {df.shape[1]} columns]\n')
        print(df.head()) 
    

def get_db_credentials(file_name):
    
    credentials = yaml.safe_load(Path(file_name).read_text())
    return credentials


def extract_rds_data(rds_table, credentials_file):
    
    credentials = get_db_credentials(credentials_file)

    connector = RDSDatabaseConnector(rds_table, credentials)
    connector.initialise_sqlalchemy_engine()
    connector.get_dataframe()
    connector.save_data_in_csv()
    connector.load_dataframe()


if __name__ == "__main__":

    extract_rds_data("customer_activity", "credentials.yaml")

