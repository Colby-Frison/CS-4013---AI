import numpy as np
import matplotlib.pyplot as plt
import os


# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# ......
# --- end of task --- #

# Create Figures directory if it doesn't exist
os.makedirs('Figures', exist_ok=True)


# load a data set for classification 
# in array "data", each row represents a patient 
# each column represents an attribute of patients 
# last column is the binary label: 1 means the patient has diabetes, 0 means otherwise
data = np.loadtxt('diabetes.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]


# --- Your Task --- #
# now, vary the percentage of data used for training 
# pick 8 values for array "num_train_per" e.g., 0.5 means using 50% of the available data for training 
# You should aim to observe overiftting (and normal performance) from these 8 values 
# Note: maximum percentage is 0.75
num_train_per = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
# --- end of task --- #

er_train_per = []
er_test_per = []
for per in num_train_per: 

    # create training data and label 
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]
    
    # we will use logistic regression model with stratification
    # Using stratified sampling to ensure consistent class distribution
    np.random.seed(42)  # Set seed for reproducibility
    
    # Get indices for each class in training data
    train_indices = np.arange(num_train)
    class_0_indices = train_indices[label_train == 0]
    class_1_indices = train_indices[label_train == 1]
    
    # Ensure we have at least one sample from each class
    if len(class_0_indices) > 0 and len(class_1_indices) > 0:
        # Calculate proportion of each class
        prop_class_1 = len(class_1_indices) / num_train
        
        # Sample from each class proportionally
        n_class_1 = max(1, int(num_train * prop_class_1))
        n_class_0 = num_train - n_class_1
        
        # Sample indices
        if len(class_0_indices) >= n_class_0 and len(class_1_indices) >= n_class_1:
            sampled_class_0 = np.random.choice(class_0_indices, size=n_class_0, replace=False)
            sampled_class_1 = np.random.choice(class_1_indices, size=n_class_1, replace=False)
            
            # Combine and shuffle
            sampled_indices = np.concatenate([sampled_class_0, sampled_class_1])
            np.random.shuffle(sampled_indices)
            
            # Update training data
            sample_train = sample_train[sampled_indices]
            label_train = label_train[sampled_indices]
    
    # Use logistic regression with consistent parameters
    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
    
    # --- Your Task --- #
    # now, training your model using training data 
    # (sample_train, label_train)
    model.fit(sample_train, label_train)

    # now, evaluate training error (not MSE) of your model 
    # store it in "er_train"
    train_predictions = model.predict(sample_train)
    train_accuracy = accuracy_score(label_train, train_predictions)
    er_train = 1 - train_accuracy  # Classification error = 1 - accuracy
    er_train_per.append(er_train)
    
    # now, evaluate testing error (not MSE) of your model 
    # store it in "er_test"
    test_predictions = model.predict(sample_test)
    test_accuracy = accuracy_score(label_test, test_predictions)
    er_test = 1 - test_accuracy  # Classification error = 1 - accuracy
    er_test_per.append(er_test)
    # --- end of task --- #
    
plt.figure(figsize=(8, 6))    
plt.plot(num_train_per,er_train_per, label='Training Error', marker='o')
plt.plot(num_train_per,er_test_per, label='Testing Error', marker='s')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Error')
plt.title('Task 4: Impact of Training Data Size on Classification')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figures/Figure3.png', dpi=300, bbox_inches='tight')
print("Figure 3 saved to Figures/Figure3.png")
plt.show()



