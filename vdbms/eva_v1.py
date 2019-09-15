"""
This file will ultimately replace the "pipeline.py" file.
The purpose of this file is to create instances of all modules, perform the final layer of abstraction for vdbms.
The basic structure of our vdbms named EVA (Exploratory Video Analytics) system is derived from work by Yao et al,
Accelerating Machine Learning Inference with Probabilistic Predicates
https://www.microsoft.com/en-us/research/blog/probabilistic-predicates-to-accelerate-inference-queries/

TODO: This example is not complete!!!

"
"""

import numpy as np
import constants

from loaders.loader_uadetrac import LoaderUADetrac
from filters.minimum_filter import FilterMinimum
from query_optimizer.qo_minimum import QOMinimum


#TODO: Fill this file in with the components loaded from other files
class Eva:
  """1. Load the dataset
     2. Load the QO
     3. Load the Filters
     4. Load the Central Network (RFCNN, SVM for other labels etc)
     5. Listen to Queries
     6. Give back result"""

  def __init__(self):
    self.LOAD = LoaderUADetrac()
    self.PP = FilterMinimum()
    self.QO = QOMinimum()
    self.image_matrix_train = None
    self.image_matrix_test = None
    self.data_table_train = None
    self.data_table_test = None
    self.filters = {} ## TODO: create multiple instances of filters for predicates

  def run(self):
    image_matrix = self.LOAD.load_images()
    labels = self.LOAD.load_labels()

    self.image_matrix_train, self.image_matrix_test, self.data_table_train, self.data_table_test = self._split_train_val(image_matrix, labels)
    self.train()
    stats = self.PP.getAllStats()
    self.execute(stats)


  def _split_train_val(self, X, label_dict):
    n_samples, _, _, _= X.shape
    mixed_indices = np.random.permutation(n_samples)
    train_index_end = int(len(mixed_indices) * 0.8)

    X_train = X[mixed_indices[:train_index_end]]
    X_test = X[mixed_indices[train_index_end:]]


    label_dict_train = {}
    label_dict_test = {}
    for column in label_dict:
        label_dict_train[column] = label_dict[column][mixed_indices[:train_index_end]]
        label_dict_test[column] = label_dict[column][mixed_indices[train_index_end:]]

    return X_train, X_test, label_dict_train, label_dict_test



  def train(self):
    """
    Need to train the PPs and UDF
    :return: trained PPs and UDF
    """
    ## TODO: Need to implement this function!!!
    """
    Filter module has changed 
    Previous: There is one filter module that trains for all the possible predicates
    Now: There are multiple filter modules and each one deals with its corresponding predicate
    Also we expect the input label to the Filters to be in binary
    """
    synthetic_pp_list = ["t=Sedan"]


  def execute(self, pp_category_stats, pp_category_models):
    TRAF_20 = ["t=van", "s>60",
                "c=white", "c!=white", "o=pt211", "c=white && t=van",
                "s>60 && s<65", "t=car || t=others", "i=pt335 && o=pt211",
                "t=van && c!=white", "c=white && t!=van && t!=car",
                "t=van && s>60 && s<65", "t=car || t=others && c!=white",
                "i=pt335 && o!=pt211 && o!=pt208", "t=van && i=pt335 && o=pt211",
                "t!=car && c!=black && c!=silver && t!=others",
                "t=van && s>60 && s<65 && o=pt211", "t!=sedan && t!=van && c!=red && t!=white",
                "i=pt335 || i=pt342 && o!=pt211 && o!=pt208",
                "i=pt335 && o=pt211 && t=van && c=red"]


    synthetic_pp_list = ["t=car", "t=bus", "t=van", "t=others",
                         "c=red", "c=white", "c=black", "c=silver",
                         "s>40", "s>50", "s>60", "s<65", "s<70",
                         "i=pt335", "i=pt211", "i=pt342", "i=pt208",
                         "o=pt335", "o=pt211", "o=pt342", "o=pt208"]


    label_desc = {"t": [constants.DISCRETE, ["car", "others", "bus", "van"]],
                  "s": [constants.CONTINUOUS, [40, 50, 60, 65, 70]],
                  "c": [constants.DISCRETE, ["white", "red", "black", "silver"]],
                  "i": [constants.DISCRETE, ["pt335", "pt342", "pt211", "pt208"]],
                  "o": [constants.DISCRETE, ["pt335", "pt342", "pt211", "pt208"]]}


    query_plans = []
    for query in TRAF_20:
    #TODO: After running the query optimizer, we want the list of PPs to work with
    #TODO: Then we want to execute the queries with the PPs and send it to the UDF after
      best_query, best_operators, reduction_rate = self.QO.run(query, synthetic_pp_list, pp_category_stats, label_desc)
    #TODO: Assume the best_query is in the form ["(PP_name, model_name) , (PP_name, model_name), (PP_name, model_name), (PP_name, model_name), (UDF_name, model_name - None)]
    #                                   operators will be [np.logical_and, np.logical_or, np.logical_and.....]
      if __debug__:
        print(("The total reduction rate associated with the query is " + str(reduction_rate)))
        print(("The best alternative for " + query + " is " + str(best_query)))
        print(("The operators involved are " + str(best_operators)))
      y_hat1 = []
      y_hat2 = []
      for i in range(len(best_query)):
        pp_name, model_name = best_query[i]
        if y_hat1 == []:
          y_hat1 = self.PP.predict(self.image_matrix_test, pp_name, model_name)
          continue
        else:
          y_hat2 = self.PP.predict(self.image_matrix_test, pp_name, model_name)
          y_hat1 = best_operators[i - 1](y_hat1, y_hat2)

      print(("The final boolean array to pass to udf is : \n" + str(y_hat1)))

      if "t=" in query and query in self.data_table_test:
        resulting_labels = self.pass_to_udf(y_hat1, query.replace("t=", ""))
        print(("Total score for this query is " + str(np.sum(resulting_labels==self.data_table_test[query]) / len(resulting_labels)) ))
      else:
        print(("No existing udf for this query: " + query))



if __name__ == "__main__":
    pipeline = Eva()
    pipeline.run()

