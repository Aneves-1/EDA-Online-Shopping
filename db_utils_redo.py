import yaml
from sqlalchemy import create_engine
from pathlib import Path
import pandas as pd


class RDSDatabaseConnector:
    """
    A class for connecting to and interacting with a PostgreSQL database on Amazon's AWS RDS.
    
    Parameters:
    -----------
    rds_table: str
        The name of the RDS table whose data will be extracted for exploratory data analysis
    db_credentials: dict
        A dictionary that is used to pass RDS database credentials, namely HOST, USER, PASSWORD, DATABASE and PORT.

    Methods:
    --------
    initialise_sqlalchemy_engine()
        Initialises the SQLAlchemy engine for database connection
    
    get_dataframe()
        Extracts an RDS database table using the SQLAlchemy engine and returns Pandas DataFrame

    save_data_in_csv()
        Saves the data extracted from RDS to the local machine in CSV format

    load_dataframe()
        Loads saved CSV file into Pandas DataFrame, prints DataFrame size in rows and columns, and the DataFrame
    """
    

    def __init__(self, rds_table: str, db_credentials: dict) -> None:
        
        self.db_credentials = db_credentials  
        self.rds_table = rds_table 
        self.csv_file_name = self.rds_table + '.csv' 

    def initialise_sqlalchemy_engine(self):
        """
        This method initialises the SQLAlchemy engine for database connection
        """

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
        """
        This method extracts an RDS database table using the SQLAlchemy engine and returns Pandas DataFrame.
        """
        
        engine = self.initialise_sqlalchemy_engine()
        dataframe = pd.read_sql_table(self.rds_table, engine)
        return dataframe


    def save_data_in_csv(self):
        """
        This method saves the data extracted from RDS to the local machine in CSV format
        """

        dataframe = self.get_dataframe()
        try:
          dataframe.to_csv(self.csv_file_name, encoding='utf-8', index=False)
          print(f"Extracted data has been successfully saved to the application directory with file name: {self.csv_file_name}.\n")
        except RuntimeError:
          print("Something went wrong.\n")


    def load_dataframe(self):
        """
        This method loads saved CSV file into Pandas DataFrame, prints DataFrame size in rows and columns, and the DataFrame        
        """

        df = pd.read_csv(Path(self.csv_file_name))
        print(f'Size of DataFrame: [{df.shape[0]} rows x {df.shape[1]} columns]\n')
        print(df.head()) 
    

def get_db_credentials(file_name):
    """
    This function gets the credentials for connecting to the remote database
    
    Parameters:
    -----------
    file_name: str
       The name of the YAML file that contains the credentials for connecting to the remote database

    Returns:
    --------
       credentials
           Data dictionary contained within the YAML file
    """

    credentials = yaml.safe_load(Path(file_name).read_text())
    return credentials


def extract_rds_data(rds_table, credentials_file):
    """
    This function creates an instance of the RDSDatabaseConnector class object and calls the relevant methods for:
    
    Parameters:
    -----------
    rds_table: str
       The name of the RDS table name whose data will be extracted
    credentials_file: str
       The name of the YAML file that contains the RDS database credentials
    """

    credentials = get_db_credentials(credentials_file)

    connector = RDSDatabaseConnector(rds_table, credentials)
    connector.initialise_sqlalchemy_engine()
    connector.get_dataframe()
    connector.save_data_in_csv()
    connector.load_dataframe()


if __name__ == "__main__":

    extract_rds_data("customer_activity", "credentials.yaml")

