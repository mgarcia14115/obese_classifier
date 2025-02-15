import uafscs.utils.data_utils    as dutils
from uafscs.utils.console_utils import get_console_args
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from uafscs.models.baseline import clf






def main():
    args = get_console_args()

    file_path     = args.file_path
    drop_list     = args.drop_features
    keep_list     = args.keep_features
    label         = args.label
    if drop_list == None and keep_list == None:
        print("You must enter a keep_features or drop_features argument")
        return;
    elif drop_list == None and keep_list != None:
        df = pd.read_csv(file_path)
        drop_list = [feature for feature in df.columns if feature not in keep_list]
    else:
        pass


    data_obj = dutils.Dataset(filepath=file_path,dropList=drop_list,label=label)

    data_obj.transform_categorical()
    data_obj.scale_data()
    data_obj.encode_labels()

    Y = data_obj.labels
    X = np.asarray(data_obj.data)

    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=42)


    print(f"Training model ... ")
    clf.fit(X_train,y_train)


    print(f"Testing model ... ")
    results = clf.score(X_test,y_test)

    print(f"{round(results*100,1)}% accuracy")

if __name__ == "__main__":
    main()