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

# Use 50% of data for training for hyperparameter analysis
per = 0.5
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]

# Hyperparameter: class weight multiplier
# This controls how much emphasis we give to the minority class
weight_multipliers = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]

auc_scores = []
acc_scores = []

for multiplier in weight_multipliers:
    
    # Calculate class weights with the hyperparameter
    class_0_count = np.sum(label_train == 0)
    class_1_count = np.sum(label_train == 1)
    
    if class_1_count > 0:
        imbalance_ratio = class_0_count / class_1_count
        weight_for_0 = 1.0
        weight_for_1 = imbalance_ratio * multiplier  # Use hyperparameter here
        class_weights = {0: weight_for_0, 1: weight_for_1}
    else:
        class_weights = None
    
    # Train model with the hyperparameter
    model = LogisticRegression(
        class_weight=class_weights,
        C=1.0,
        random_state=42,
        max_iter=1000
    )
    model.fit(sample_train, label_train)
    
    # Evaluate performance
    test_predictions = model.predict(sample_test)
    test_proba = model.predict_proba(sample_test)[:, 1]
    
    acc = accuracy_score(label_test, test_predictions)
    auc = roc_auc_score(label_test, test_proba)
    
    acc_scores.append(acc)
    auc_scores.append(auc)
    
    print(f"Weight Multiplier: {multiplier:.1f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")

# Create Figure 6: Impact of Class Weight Multiplier on AUC
plt.figure(figsize=(10, 6))
plt.plot(weight_multipliers, auc_scores, 'o-', linewidth=2, markersize=8, label='AUC Score', color='blue')
plt.plot(weight_multipliers, acc_scores, 's-', linewidth=2, markersize=8, label='Accuracy', color='red')
plt.xlabel('Class Weight Multiplier (Hyperparameter)', fontsize=12)
plt.ylabel('Performance Score', fontsize=12)
plt.title('Figure 6: Impact of Class Weight Multiplier on Model Performance', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xscale('log')  # Use log scale for better visualization
plt.tight_layout()
plt.savefig('Figures/Figure6.png', dpi=300, bbox_inches='tight')
print("Figure 6 saved to Figures/Figure6.png")
plt.show()

# Find optimal hyperparameter
optimal_idx = np.argmax(auc_scores)
optimal_multiplier = weight_multipliers[optimal_idx]
optimal_auc = auc_scores[optimal_idx]
optimal_acc = acc_scores[optimal_idx]

print(f"\nOptimal Class Weight Multiplier: {optimal_multiplier:.1f}")
print(f"Optimal AUC Score: {optimal_auc:.4f}")
print(f"Optimal Accuracy: {optimal_acc:.4f}")

# Show the impact range
auc_range = max(auc_scores) - min(auc_scores)
print(f"AUC Score Range: {auc_range:.4f} (from {min(auc_scores):.4f} to {max(auc_scores):.4f})")
