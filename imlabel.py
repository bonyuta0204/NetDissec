import pandas as pd

label = pd.read_csv("./imalabel.csv", sep=";", index_col=0)
print(label.ix[0, 4])
