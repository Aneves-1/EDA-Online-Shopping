# Project Title
    Exploratory Data Analysis - Online Shopping in Retail
    
## Project description
     This project is part of the data analysis training provided by AICore, using data to provide a realife mimic of a data analyst approach to real life business needs.
     With it, via python, I've learned to get files from AWS, cleaned and transformed the data for analysis, and the performed exploratory data analysis to answer the marketing team questions.
     Within the EDA, I've struggle and learned with the best methods to reduce skew, replace null and remove outliers, as within the marketing team questions, various methos to affect lead to various different interpretations of the data from a business POV.
   
## Installation instructions
    Run via VSCode, requiring python, jupyter, pandas, SQLAlchemy, Git, Amazon AWS

## Usage instructions
    1 - Run db_utils.py to extract the data from AWS and save it into a csv file. (file already included on repository since credentials are confidential)
    2 - Run EDA jupyter file, which will take through the data transformation and the data analysis

## File structure of the project
    .gitignore 
    EDA.ipynb - jupyter notebook with the EDA transformation process and business analysis
    README.md
    customer_activity.csv - data to process
    datatransform.py - scripts to be used at EDA for visualisation and transformation
    bd_utils.py - script to extract data from AWS
    
## License information
    Dataset obtained via AWS RDS through AICore, which also provided the credentails to download the data set. This is an open source public repository.
