import numpy as np
import matplotlib.pyplot as plt
import os

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
# ......
# --- end of task --- #

# Create Figures directory if it doesn't exist
os.makedirs('Figures', exist_ok=True)

# load a data set for regression
# in array "data", each row represents a community 
# each column represents an attribute of community 
# last column is the continuous label of crime rate in the community
data = np.loadtxt('crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]


# --- Your Task --- #
# now, pick the percentage of data used for training 
# remember we should be able to observe overfitting with this pick 
# note: maximum percentage is 0.75 
per = 0.15
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 8 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []
for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #
    # now train your model using (sample_train, label_train)
    model.fit(sample_train, label_train)
    # now evaluate your training error (MSE) and stores it in "er_train"
    train_predictions = model.predict(sample_train)
    er_train = mean_squared_error(label_train, train_predictions)
    er_train_alpha.append(er_train)
    # now evaluate your testing error (MSE) and stores it in "er_test"
    test_predictions = model.predict(sample_test)
    er_test = mean_squared_error(label_test, test_predictions)
    er_test_alpha.append(er_test)
    # --- end of task --- #

    
plt.figure(figsize=(8, 6))
plt.plot(alpha_vec,er_train_alpha, label='Training Error', marker='o')
plt.plot(alpha_vec,er_test_alpha, label='Testing Error', marker='s')
plt.xlabel('Hyper-Parameter Alpha')
plt.ylabel('Prediction Error (MSE)')
plt.title('Task 2: Impact of Hyper-Parameter on Regression')
plt.xscale('log')  # Use log scale for alpha values
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figures/Figure2.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved to Figures/Figure2.png")
plt.show()


