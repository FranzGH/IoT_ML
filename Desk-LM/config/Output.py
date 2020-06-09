import json
import jsonschema
from jsonschema import validate

import os

import numpy as np

import error as error

# Describe what kind of json you expect.
outputSchema = {
    "type": "object",
    "properties": {
        "export_path": {"type": "string"},
        "is_dataset_test": {"type": "boolean"},
        "dataset_test_size": {"type": "number"}
    },
    #"required": ["export_path"],
    "additionalProperties": False
}

class Output(object):

    # Constructor
    def __init__(self, jsonFilePath):
        try:
            with open(jsonFilePath) as json_file:
                try:
                    jsonData = json.load(json_file)            
                    validate(instance=jsonData, schema=outputSchema)
                except jsonschema.exceptions.ValidationError as err:
                    print(err)
                    raise ValueError(error.errors['output_config'])
                except ValueError as err:
                    print(err)
                    raise ValueError(error.errors['output_config'])
                self.parse(jsonData)
        except FileNotFoundError as err:
                print(err)
                raise ValueError(error.errors['output_config'])

    def parse(self, jsonData):
        try:
            self.export_path = None
            self.is_dataset_test = False
            self.dataset_test_size = 1
            if 'export_path' in jsonData:
                self.export_path = jsonData['export_path']
            if 'is_dataset_test' in jsonData:
                self.is_dataset_test = jsonData['is_dataset_test']
            if 'dataset_test_size' in jsonData:
                self.dataset_test_size = jsonData['dataset_test_size']
        except Exception as e:
            print(e)
            raise ValueError(error.errors['output_config'])