import json
import jsonschema
from jsonschema import validate

import abc

import error as error

# Describe what kind of json you expect.
estimatorSchema = {
    "type": "object",
    "properties": {
        "estimator": {"type": "string"}
    },
    "required": ["estimator"]
}

class Estimator(object):

    @staticmethod
    def create(jsonFilePath, dataset):
        try:
            with open(jsonFilePath) as json_file:
                try:
                    jsonData = json.load(json_file)           
                    validate(instance=jsonData, schema=estimatorSchema)
                except jsonschema.exceptions.ValidationError as err:
                    print(err)
                    raise ValueError(error.errors['estimator_config'])
                except ValueError as err:
                    print(err)
                    raise ValueError(error.errors['estimator_config'])
                
                if jsonData['estimator'].startswith('KNeighbors'):                
                    import Knn #as Knn
                    esti = Knn.Knn(jsonData)
                elif jsonData['estimator'].startswith('DecisionTree'):                
                    import DecisionTree
                    esti = DecisionTree.DecisionTree(jsonData)
                else:
                    est_str = jsonData['estimator']
                    print(f'Invalid value for estimator name: {est_str}')
                    raise ValueError(error.errors['estimator_config'])
                esti.parse(jsonData)
                esti.assign_dataset(dataset)
                return esti
        except FileNotFoundError as err:
                print(err)
                raise ValueError(error.errors['estimator_config'])

    def assign_dataset(self, dataset):
        self.dataset = dataset
        if not self.is_regr:
            self.n_classes = self.dataset.y.nunique()
            if self.n_classes == 1:
                self.n_classes = 2

'''
    # Constructor
    def __init__(self, jsonFileName):
        try:
            with open(jsonFileName) as json_file:
                jsonData = json.load(json_file)           
                try:
                    validate(instance=jsonData, schema=estimatorSchema)
                except jsonschema.exceptions.ValidationError as err:
                    print(err)
                    raise ValueError(error.errors['estimator_config'])
                self.parse(jsonData)
        except FileNotFoundError as err:
                print(err)
                raise ValueError(error.errors['estimator_config'])

    def parse(self, jsonData):
        if(jsonData['estimator']=='KNeighborsClassifier'):
            from sklearn.neighbors import KNeighborsClassifier
            self.estimator = KNeighborsClassifier()
            self.is_class = True
            self.params = {}
            if "n_neighbors_array" in jsonData:
                self.params['n_neighbors'] = jsonData['n_neighbors_array']
            elif "n_neighbors" in jsonData:
                self.params['n_neighbors'] = jsonData['n_neighbors']
            else:
                if "n_neighbors_lowerlimit" in jsonData:
                    l = jsonData['n_neighbors_lowerlimit']
                else:
                    l = 1
                if "n_neighbors_upperlimit" in jsonData:
                    u = jsonData['n_neighbors_upperlimit']
                else:
                    u = 5
                if "n_neighbors_interval" in jsonData:
                    import numpy as np
                    i = jsonData['n_neighbors_interval']
                    self.params['n_neighbors'] = np.arange(l, u, i)
                else:
                    self.params['n_neighbors'] = np.arange(l, u)
            sys.path.insert(1, 'output')
            import Knn_OM as Knn_OM
            self.output_manager = Knn_OM.Knn_OM()
        else:
            est_str = jsonData['estimator']
            print(f'Invalid value for estimator name: {est_str}')
            raise ValueError(error.errors['estimator_config'])
'''