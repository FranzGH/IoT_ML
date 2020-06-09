import json
import jsonschema
from jsonschema import validate

import os

import numpy as np

import error as error

# Describe what kind of json you expect.
datasetSchema = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "skip_rows": {"type": "number"},
        "skip_columns": {
            "type": "array",
            "items": {
                "type": "string"
            }
            },
        "target_column": {"type": "string"},
        "test_size": {"type": "number"}
    },
    "required": ["path"],
    "additionalProperties": False
}

class Dataset(object):

    # Constructor
    def __init__(self, jsonFilePath):
        try:
            with open(jsonFilePath) as json_file:
                try:
                    jsonData = json.load(json_file)            
                    validate(instance=jsonData, schema=datasetSchema)
                except jsonschema.exceptions.ValidationError as err:
                    print(err)
                    raise ValueError(error.errors['ds_config'])
                except ValueError as err:
                    print(err)
                    raise ValueError(error.errors['ds_config'])
                self.parse(jsonData)
                base = os.path.basename(jsonFilePath)
                self.name = os.path.splitext(base)[0]
        except FileNotFoundError as err:
                print(err)
                raise ValueError(error.errors['ds_config'])

    def parse(self, jsonData):
        try:
            import pandas as pd
            skiprows=0
            if 'skip_rows' in jsonData:
                skiprows = jsonData['skip_rows']
            self.df = pd.read_csv(jsonData['path'], skiprows=skiprows)
            if 'skip_columns' in jsonData:
                self.df.drop(jsonData['skip_columns'], axis = 1, inplace=True)
            if 'target_column' in jsonData:
                self.y = self.df.loc[:,jsonData['target_column']]
                self.X = self.df.drop(jsonData['target_column'], axis = 1)
            else:
                self.y = self.df.iloc[:,-1]
                self.X = self.df.drop(self.df.columns[-1], axis = 1)
            
            #self.X.replace(r'^\s*$', np.nan, regex=True, inplace=True)           
            self.X = self.X.apply(lambda x: x.fillna(x.mean()),axis=0)

            if 'test_size' in jsonData:
                self.test_size = jsonData['test_size']
            else:
                self.test_size = 0.3
        except Exception as e:
            print(e)
            raise ValueError(error.errors['ds_config'])