from pandas import get_dummies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def prepare_data(dataset):
    return get_dummies(dataset, columns=['University_Rating','Research'])

def standardize(x):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x)
 
def train_test_splitter(x_scaled, AdmitChance):
    # Split the dataset using stratified sampling to ensure both classes are split in good proportion
    x_train, x_test, y_train, y_test = train_test_split(x_scaled,AdmitChance, test_size=0.2, random_state=123, stratify=AdmitChance)
    
    print("Train test split shape: ")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test
