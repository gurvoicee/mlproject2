import os
import pandas as pd

from src.utils import load_object
from sklearn.model_selection import train_test_split

features = {
                "RM": 6.012,
                "LSTAT": 12.43,
                "PTRATIO": 15.2
            }

df =  pd.DataFrame(features,index = [0])

model_path=os.path.join("artifacts","model.pkl")
preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
print("Before Loading")
model=load_object(file_path=model_path)
preprocessor=load_object(file_path=preprocessor_path)
print("After Loading")


array = df.to_numpy()
reshaped_array = array.reshape(1, -1)
reshaped_df = pd.DataFrame(reshaped_array, columns=df.columns)

data_scaled=preprocessor.transform(reshaped_df)
preds=model.predict(data_scaled)
print(preds[0])