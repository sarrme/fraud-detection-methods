import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras import layers



# confusion matrix
def cf_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# accuracy and precision values 
def clf_report(y_test, y_pred):
    print("Accuracy : ", accuracy_score(y_test, y_pred))   
    print("Precision :",  precision_score(y_test, y_pred)) 
    print("Recall :",  recall_score(y_test, y_pred)) 
    print("F1 : ", f1_score(y_test, y_pred)) 

# split data into train and test 
def split(df, test_size=0.2, random_state=0): 
    features_cols = ["Amount"] + [col for col in df.columns if col.startswith("V")]

    X = df[features_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state) 


    return X_train, X_test, y_train, y_test 


# plot the loss values for deep learning model
def loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("loss.png")


# construct a network with the same input and output size 
def network(dims=[], input_dim=29, activation="elu"):

    assert len(dims) > 0

    input = keras.Input(shape=(input_dim,))
    layer = input 
    for dim in dims:
        layer = layers.Dense(dim, activation=activation)(layer)
    decoded = layers.Dense(input_dim)(layer)

    autoencoder = keras.Model(input, decoded)
    return autoencoder