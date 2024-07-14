# Import libraries
import pandas as pd
import numpy as np


def import_data(path_to_file):
    input_df = pd.read_csv(path_to_file)
    return input_df
