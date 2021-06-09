from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore") #supress warnings

headers = ['age', 'sex','chest_pain','resting_blood_pressure',  
        'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
        'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
        'num_of_major_vessels','thal', 'heart_disease']

heart_df = pd.read_csv('heart.dat', sep=' ', names=headers)

x = heart_df.drop(columns=['heart_disease'])

heart_df['heart_disease'] = heart_df['heart_disease'].replace(1,0)
heart_df['heart_disease'] = heart_df['heart_disease'].replace(2,1)

y = heart_df['heart_disease'].values.reshape(x.shape[0], 1)

#split data into train and test sets
x_train, x_test = train_test_split(x, test_size=0.2, random_state=2, shuffle=False)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=2, shuffle=False)


#standardize the dataset
sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)


sknet = MLPClassifier(hidden_layer_sizes=8, learning_rate_init=0.001, max_iter=100)

sknet.fit(x_train, y_train)
predict_train = sknet.predict(x_train)
predict_test = sknet.predict(x_test)

print("Train accuracy of sklearn neural network: {}".format(round(accuracy_score(predict_train, y_train),2)*100))
print("Test accuracy of sklearn neural network: {}".format(round(accuracy_score(predict_test, y_test),2)*100))