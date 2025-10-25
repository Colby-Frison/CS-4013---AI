import numpy as np
import matplotlib.pyplot as plt
import os

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
# ......
# --- end of task --- #

# Create Figures directory if it doesn't exist
os.makedirs('Figures', exist_ok=True)

# load an imbalanced data set 
# there are 50 positive class instances 
# there are 500 negative class instances 
data = np.loadtxt('diabetes_new.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# vary the percentage of data for training
num_train_per = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

acc_base_per = []
auc_base_per = []

acc_yours_per = []
auc_yours_per = []

for per in num_train_per: 

    # create training data and label
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]

    model = LogisticRegression()

    # --- Your Task --- #
    # Implement a baseline method that standardly trains 
    # the model using sample_train and label_train
    model.fit(sample_train, label_train)
    
    # evaluate model testing accuracy and stores it in "acc_base"
    test_predictions = model.predict(sample_test)
    acc_base = accuracy_score(label_test, test_predictions)
    acc_base_per.append(acc_base)
    
    # evaluate model testing AUC score and stores it in "auc_base"
    test_proba = model.predict_proba(sample_test)[:, 1]
    auc_base = roc_auc_score(label_test, test_proba)
    auc_base_per.append(auc_base)
    # --- end of task --- #
    
    
    # --- Your Task --- #
    # Now, implement your method 
    # Aim to improve AUC score of baseline 
    # while maintaining accuracy as much as possible 
    
    # Method: Enhanced class weighting with regularization tuning
    # This approach uses stronger class weighting and optimized regularization
    
    # Calculate class weights with stronger emphasis on minority class
    class_0_count = np.sum(label_train == 0)
    class_1_count = np.sum(label_train == 1)
    
    if class_1_count > 0:
        # Use moderate class weighting to emphasize minority class
        imbalance_ratio = class_0_count / class_1_count
        # Use a more moderate multiplier to balance AUC improvement and accuracy
        weight_for_0 = 1.0
        weight_for_1 = imbalance_ratio * 0.8  # 0.8x multiplier for moderate emphasis
        class_weights = {0: weight_for_0, 1: weight_for_1}
    else:
        class_weights = None
    
    # Train model with enhanced class weights and moderate regularization
    model_improved = LogisticRegression(
        class_weight=class_weights,
        C=1.0,  # Standard regularization
        random_state=42,
        max_iter=1000
    )
    model_improved.fit(sample_train, label_train)
    
    # Evaluate model testing accuracy and store it in "acc_yours"
    test_predictions = model_improved.predict(sample_test)
    acc_yours = accuracy_score(label_test, test_predictions)
    acc_yours_per.append(acc_yours)
    
    # Evaluate model testing AUC score and store it in "auc_yours"
    test_proba = model_improved.predict_proba(sample_test)[:, 1]
    auc_yours = roc_auc_score(label_test, test_proba)
    auc_yours_per.append(auc_yours)
    # --- end of task --- #
    

plt.figure(figsize=(8, 6))    
plt.plot(num_train_per,acc_base_per, label='Base Accuracy', marker='o')
plt.plot(num_train_per,acc_yours_per, label='Your Accuracy', marker='s')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Accuracy')
plt.title('Task 5: Model Accuracy vs Training Data Size (Imbalanced Data)')
plt.ylim(0.65, 1.0)  # Set y-axis bounds for better visualization
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figures/Figure4.png', dpi=300, bbox_inches='tight')
print("Figure 4 saved to Figures/Figure4.png")
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(num_train_per,auc_base_per, label='Base AUC Score', marker='o')
plt.plot(num_train_per,auc_yours_per, label='Your AUC Score', marker='s')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification AUC Score')
plt.title('Task 5: Model AUC Score vs Training Data Size (Imbalanced Data)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figures/Figure5.png', dpi=300, bbox_inches='tight')
print("Figure 5 saved to Figures/Figure5.png")
plt.show()


