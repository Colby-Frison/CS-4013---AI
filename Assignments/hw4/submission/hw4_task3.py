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
per = 0.5
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [0.01, 0.1, 1, 10, 100]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []
for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #
    # now implement k-fold cross validation 
    # on the training set (which means splitting 
    # training set into k-folds) to get the 
    # validation error for each candidate alpha value 
    # store it in "er_valid"
    k = 5
    fold_size = num_train // k
    er_valid_folds = []
    for fold in range(k):
        # define validation set for this fold
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k-1 else num_train

        # split data into train and validation
        val_indices = list(range(val_start, val_end))
        train_indices = list(range(0, val_start)) + list(range(val_end, num_train))

        # create training and validation sets
        fold_train_sample = sample_train[train_indices]
        fold_train_label = label_train[train_indices]
        fold_val_sample = sample_train[val_indices]
        fold_val_label = label_train[val_indices]

        # train model on this fold
        model.fit(fold_train_sample, fold_train_label)
        pred_val = model.predict(fold_val_sample)
        er_val = mean_squared_error(fold_val_label, pred_val)
        er_valid_folds.append(er_val)
    
    # average validation error across all folds
    er_valid = np.mean(er_valid_folds)
    er_train_alpha.append(er_valid) 
    # --- end of task --- #


# Now you should have obtained a validation error for each alpha value 
# In the homework, you just need to report these values

# Print Table 1 for the homework
print("\n" + "="*60)
print("Table 1: K-Fold Cross-Validation Results")
print("="*60)
print(f"{'Alpha Value':<20} {'Validation Error':<20}")
print("-"*60)
for alpha, er_valid in zip(alpha_vec, er_train_alpha):
    print(f"{alpha:<20.3f} {er_valid:<20.6f}")
print("="*60)

# Save table to text file
with open('Figures/Table1.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("Table 1: K-Fold Cross-Validation Results\n")
    f.write("="*60 + "\n")
    f.write(f"{'Alpha Value':<20} {'Validation Error':<20}\n")
    f.write("-"*60 + "\n")
    for alpha, er_valid in zip(alpha_vec, er_train_alpha):
        f.write(f"{alpha:<20.3f} {er_valid:<20.6f}\n")
    f.write("="*60 + "\n")
print("Table 1 saved to Figures/Table1.txt")

# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error 
# set it to "alpha_opt"
alpha_opt = alpha_vec[np.argmin(er_train_alpha)]

# now retrain your model on the entire training set using alpha_opt 
# then evaluate your model on the testing set 
model = Ridge(alpha = alpha_opt)
model.fit(sample_train, label_train)
pred_train = model.predict(sample_train)
er_train = mean_squared_error(label_train, pred_train)
pred_test = model.predict(sample_test)
er_test = mean_squared_error(label_test, pred_test)


print(f"Alpha Opt: {alpha_opt}")
print(f"Training Error: {er_train}")
print(f"Testing Error: {er_test}")






