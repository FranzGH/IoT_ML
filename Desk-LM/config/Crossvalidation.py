import json
import jsonschema
from jsonschema import validate

import error as error

# Describe what kind of json you expect.
cvSchema = {
    "type": "object",
    "properties": {
        "cv": {
            "type": "number"
            },
        "scoring": {
            "type": "string"
            }
    },
    # "required": ["cv"],
    "additionalProperties": False
}

class Crossvalidation(object):

    # Constructor
    def __init__(self, jsonFilePath):
        if jsonFilePath != None:
            try:
                with open(jsonFilePath) as json_file:
                    try:
                        jsonData = json.load(json_file)                               
                        validate(instance=jsonData, schema=cvSchema)
                    except jsonschema.exceptions.ValidationError as err:
                        print(err)
                        raise ValueError(error.errors['crossvalidation_config'])
                    except ValueError as err:
                        print(err)
                        raise ValueError(error.errors['crossvalidation_config'])
                    self.parse(jsonData)
            except FileNotFoundError as err:
                    print(err)
                    raise ValueError(error.errors['crossvalidation_config'])
        else:
            self.cv = None
            self.scoring = None

    def parse(self, jsonData):
        if 'cv' in jsonData:
            self.cv = jsonData['cv']
        else:
            self.cv = None
        if 'scoring' in jsonData:
            self.scoring = jsonData['scoring']
        else:
            self.scoring = None