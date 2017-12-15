import pandas as pd   
  
Images = pd.read_csv("Segmentation_DataSet/segmentation_trainning.csv") # the iris dataset is now a Pandas DataFrame(similar to excel file)

Images_Features_Dictionery = {
        1:"REGION-CENTROID-COL",
        2:"REGION-CENTROID-ROW",
        3:"REGION-PIXEL-COUNT",
        4:"SHORT-LINE-DENSITY-5",
        5:"SHORT-LINE-DENSITY-2",
        6:"VEDGE-MEAN",
        7:"VEDGE-SD",
        8:"HEDGE-MEAN",
        9:"HEDGE-SD",
        10:"INTENSITY-MEAN",
        11:"RAWRED-MEAN",
        12:"RAWBLUE-MEAN",
        13:"RAWGREEN-MEAN",
        14:"EXRED-MEAN",
        15:"EXBLUE-MEAN",
        16:"EXGREEN-MEAN",
        17:"VALUE-MEAN",
        18:"SATURATION-MEAN",
        19:"HUE-MEAN"}

label = Images["Label"]

