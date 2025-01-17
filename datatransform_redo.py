from typing import List
import numpy as np
from tabulate import tabulate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import missingno as msno
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot


class DataFrameTransform:
    """
    A class for performing various transformations on a DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame):
        
        self.df = dataframe.copy()

    def convert_to_type(self, column_name: str, data_type: str, ignore_errors: bool = True) -> pd.DataFrame:
        """
        Convert a specific column in the DataFrame to the specified data type.

        Parameters:
        - column_name (str): Name of the column to be converted.
        - data_type (str): Target data type.
        - ignore_errors (bool, optional): Whether to ignore errors during conversion. Default is True.

        """

        data_type = data_type.lower()
                
        try:
            if data_type in ["datetime", "date"]:
                self.df[column_name] = pd.to_datetime(self.df[column_name])
            elif data_type in ["str", "int", "float", "bool", "int64", "float64"]:
                data_type = data_type.replace("64", "")
                self.df[column_name] = self.df[column_name].astype(data_type)
            elif data_type == "categorical":
                self.df[column_name] = pd.Categorical(self.df[column_name])
            else:
                print(f"Error: data type {data_type} not supported. Check docstrings or call help for more information.")
        except Exception as e:
            print(f"Error converting column '{column_name}' to type '{data_type}': {e}")
             
    def convert_month_to_int(self, column_name: str) -> pd.DataFrame:
        """
        Convert a column representing months to an integer format.

        Parameters:
        - column_name (str): Name of the column to be converted.
        """

        try:
            self.df[column_name] = self.df[column_name].astype(str)
            self.df['month'] = self.df['month'].str.lower()
            month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'june': 6,
                        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            self.df[column_name] = self.df[column_name].map(month_map)
            self.df[column_name] = pd.to_numeric(self.df[column_name])
        except Exception as e:
            print(f"Error converting 'month' column to integer: {e}")
        return self.df.copy()
        
    def convert_columns(self, column_list: List[str], data_type: str, ignore_errors: bool = True) -> pd.DataFrame:
        """
        Convert multiple columns to the specified data type.

        Parameters:
        - column_list (List[str]): List of column names to be converted.
        - data_type (str): Target data type.
        - ignore_errors (bool, optional): Whether to ignore errors during conversion. Default is True.

        """
        for column in column_list:
            self.convert_to_type(column, data_type, ignore_errors)
        return self.df.copy()
        
    def impute_nulls(self, column_list: List[str], method: str) -> pd.DataFrame:
        """
        Impute null values in specified columns using a specified method.

        Parameters:
        - column_list (List[str]): List of column names to impute null values.
        - method (str): Imputation method ('mean', 'median', or 'mode').

        """

        method = method.lower()
        valid_methods = ['mean', 'median', 'mode']

        try:
            if method not in valid_methods:
                raise ValueError(f"Invalid imputation method. Method can only be one of: {', '.join(valid_methods)}")

            for column in column_list:
                if method == 'median':
                    self.df[column] = self.df[column].fillna(self.df[column].median())
                elif method == 'mean':
                    self.df[column] = self.df[column].fillna(self.df[column].mean())
                elif method == 'mode':
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0])

        except ValueError as ve:
            print(f"Error: {ve}. Please check that you have provided a list of column names formatted as strings.")
        return self.df.copy()

    def impute_nulls_with_median(self, column_list: List[str]) -> pd.DataFrame: # get rid of this since the top one is working now? WIP
        """
        Impute null values in specified columns using the median.

        Parameters:
        - column_list (List[str]): List of column names to impute null values.

        """

        for column in column_list:
            self.df[column] = self.df[column].fillna(self.df[column].median())
        return self.df
    
    def impute_nulls_with_mean(self, column_list: List[str]) -> pd.DataFrame: # ditto WIP
        """
        Impute null values in specified columns using the mean.

        Parameters:
        - column_list (List[str]): List of column names to impute null values.

        """

        for column in column_list:
            self.df[column] = self.df[column].fillna(self.df[column].mean())
        return self.df
    
    def impute_nulls_with_mode(self, column_list: List[str]) -> pd.DataFrame: # ditto WIP
        """
        Impute null values in specified columns using the mode.

        Parameters:
        - column_list (List[str]): List of column names to impute null values.

        """

        for column in column_list:
            self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        return self.df
    
    def log_transform(self, column_list: List[str]) -> pd.DataFrame:
        """
        Apply a log transformation to specified columns.

        Parameters:
        - column_list (List[str]): List of column names to transform.

        """

        for col in column_list:
            self.df[col] = self.df[col].map(lambda i: np.log(i) if i > 0 else 0)
        return self.df
    
    def yeo_johnson_transform(self, column_list: List[str]) -> pd.DataFrame:
        """
        Apply a Yeo-Johnson transformation to specified columns.

        Parameters:
        - column_list (List[str]): List of column names to transform.

        """

        for col in column_list:
            nonzero_values = self.df[col][self.df[col] != 0]
            yeojohnson_values, lambda_value = stats.yeojohnson(nonzero_values)
            self.df[col] = self.df[col].apply(lambda x: stats.yeojohnson([x], lmbda=lambda_value)[0] if x != 0 else 0)
        return self.df


class DataFrameInfo:
    """
    Initialize the DataFrameInfo object with a DataFrame. Used internally when an instance of the call is called.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to analyze.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize the DataFrameInfo object with a DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to analyze.

        """

        self.df = dataframe.copy()

    def null_counts(self, columns = None) -> pd.DataFrame:
        """
        Generate null counts in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        null_count = self.isnull().sum(axis = 0)
        null_percentage = null_count/len(self)*100
        print ("NULL COUNT")
        print (null_count)

    def null_percentage(self, columns = None) -> pd.DataFrame:
        """
        Generate null percentages in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        null_count = self.isnull().sum(axis = 0)
        null_percentage = null_count/len(self)*100
        print ("NULL %")
        print(null_percentage)

    def unique_counts_all(self, columns = None) -> pd.DataFrame:
        """
        Generate unique counts of the DataFrame.
        """

        unique_count = self.nunique()
        print ("UNIQUE COUNTS")
        print(unique_count)

    def unique_counts_select(self, columns: List[str]) -> None:
        """
        Generate unique counts in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        unique_count = self.nunique()
        print ("UNIQUE COUNTS")
        print(unique_count)

    def value_counts(self, columns = None) -> pd.DataFrame:
        """
        Generate value counts of the DataFrame.
        """
         
        value_counts = self.value_counts()
        print ("Value Counts")
        print (value_counts)

    def shape(self, columns = None) -> pd.DataFrame:
        """
        Prints shape of the DataFrame
        """

        shape = self.shape
        print ("The shape of the data is:", shape)
        
    def statistics_info(self, columns = None) -> pd.DataFrame: # needs fixing/adapted into other percentiles WIP
        """
        Generate statistical central data table of the DataFrame.
        """

        print(self.describe())
        
    def columns_info(self, columns = None) -> pd.DataFrame:
        """
        Prints .info of the DataFrame
        """
        
        print ("Column caracteristics")
        print (self.info())    

    def data_skewness_values(self, columns: List[str]) -> None:
        """
        Generate skew values in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        skew_data = []
        for col in columns:
            skew_value = self.df[col].skew()
            skew_data.append([col, skew_value])
        
        print(skew_data)

    def data_skewness_value(self): #needs adapting for list of columns to be specified WIP
        """
        Calculate and display skewness values for DataFrame.
        """
        
        skew_value = self.skew()
                
        print(skew_value)

    def z_scores(self, column: str) -> pd.DataFrame:
        
        mean_col = np.mean(self[column])
        std_col = np.std(self[column])
        z_scores = (self[column] - mean_col) / std_col
        col_values = self[[column]].copy()
        col_values['z-scores'] = z_scores
        return col_values
    
    def IQR_outlier(self, column: str) -> pd.DataFrame:
        """
        Generate IQR, Q1 and Q3 values to compare with .shape in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        Q1 = self[column].quantile(0.25)
        Q3 = self[column].quantile(0.75)
        IQR = Q3 - Q1[column]
        outliers = self.df[(self[column] < (Q1 - 1.5 * IQR)) | (self[column] > (Q3 + 1.5 * IQR))]
        print("Outliers:")
        print(f'shape: {outliers.shape}')
        print (f'Q1 = {Q1}')
        print (f'Q3 = {Q3}')
        print (f'IQR = {IQR}')


class Plotter:
    """
    A class for creating various plots and visualizations of a DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame):
        
        self.df = dataframe.copy()

    def qq_plot(self, column_list: List[str]) -> None:
        """
        Generate QQ plot in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        for column in column_list:
            qqplot(self.df[column], scale=1 ,line='q')
 
    def boxplot(self, columns: List[str]) -> None:
        """
        Generate boxplot in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        sns.boxplot(self[columns])

    def correlation_heatmap(self, column_list: List[str]) -> None:
        """
        Generate heatmap of the correlation in columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        sns.heatmap(self.df[column_list].corr())

    def nulls_heatmap(self) -> None:
        """
        Generate heatmap of nulls in the DataFrame.
        """

        msno.heatmap(self.df)

    def pair_plot(self, columns: List[str]) -> None:
        """
        Generate pair plot graph between columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        sns.pairplot(self[columns], corner=True)

    def count_plot(self, x: str, **kwargs) -> None: #WIP
        """
        Generate countplot of the DataFrame.
        """
        sns.countplot(x=x)
        x=plt.xticks(rotation=90)

    def count_plots_grid(self, columns: List[str]) -> None:  #WIP
        """
        Generate countplot of columns of a subset of the DataFrame.

        Parameters:
        - columns: Columns to include in the subset.
        """

        figure = pd.melt(self)
        grid = sns.FacetGrid(figure, col='variable',  col_wrap=3, sharex=False, sharey=False)
        grid = grid.map(self.count_plot, 'value')