# Clean kaggle housing data

# Import libraries
import sys
import pandas as pd
import statistics
import re

# Load the data
houses = pd.read_csv(sys.argv[1])

# Replace column names with bad chars
col_names = [key for key in dict(houses.dtypes) if re.search("^[^A-Z]",key) != None]
new_col_names = ['FirstFlrSF', 'SecondFlrSF', 'ThirdSsnPorch']
new_col_names = dict(zip(col_names,new_col_names))
houses.rename(columns=new_col_names, inplace=True)

# Replace missing Lot Frontage with 0
houses.LotFrontage.fillna(0, inplace=True)

# Drop useless columns: ID
# houses.drop(["Id"], axis = 1, inplace=True)

houses.to_csv(sys.argv[2], index = False)
