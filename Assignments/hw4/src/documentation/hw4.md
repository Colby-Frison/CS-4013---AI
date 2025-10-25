```markdown
# CS4013/5013 Assignment 4

Fall 2025

In this assignment, we will implement some simple machine learning processes in Python and report observations. Below is a concise description of each programming task. We will provide one template for each task. See detailed instructions in the templates.

## Part I. Regression

### Task 1
Implement the learning process of a regression model and report the impact of training data size on the model's prediction performance in **Figure 1**. Specifically:
- **x-axis**: percent of data used for training  
- **y-axis**: prediction error (mean-squared-error)  
- The figure should contain two curves: one for training error and one for testing error.

**Goal**: Observe overfitting in Figure 1.

---

### Task 2
Implement the learning process of a regression model and report the impact of hyper-parameter on the model's prediction performance in **Figure 2**. Specifically:
- **x-axis**: hyper-parameter value  
- **y-axis**: prediction error  
- The figure should contain two curves: one for training error and one for testing error.

**Goal**: Observe both overfitting and underfitting in Figure 2.

---

### Task 3
Implement the k-fold cross-validation technique and apply it to select an optimal hyper-parameter for a regression model. (*Do not call a cross-validation function; implement data splitting, training, and evaluation yourself.*)  
Pick 5 candidate values for the hyper-parameter and report the k-fold cross-validation error of each value in **Table 1**.

**Goal**: The optimal error should occur when the hyper-parameter is neither too small nor too large.

---

## Part II. Classification

### Task 4
Implement the learning process of a classification model and report the impact of training data size on the model's prediction performance in **Figure 3**. Specifically:
- **x-axis**: percentage of data used for training  
- **y-axis**: prediction error (classification error, not MSE)  
- The figure should contain two curves: one for training error and one for testing error.

**Goal**: Observe overfitting in Figure 3.

---

### Task 5
Implement a learning process for a classification model on an unbalanced dataset and evaluate model performance (both classification error and AUC score).

Develop your own method to improve the AUC score while maintaining classification error as much as possible.  
*A useful reference: "Haibo He and Edwardo A. Garcia, Learning from Imbalanced Data, IEEE Transactions on Knowledge and Data Engineering, 2009" (PDF provided in the assignment folder).*

Report testing performance of both the baseline method and your method in **Figure 4** and **Figure 5**:
- **Figure 4**: model accuracy vs. training data size  
- **Figure 5**: model AUC score vs. training data size

**Goal**:
- Your method's AUC curve should be higher than the baseline's curve.
- Your method's accuracy curve should be as close to the baseline's curve as possible.

**Bonus Point**: If your method has a hyper-parameter and you can show a figure (**Figure 6**) demonstrating its (reasonable) impact on your model's AUC score, you get a bonus point.

---

## Submissions Instructions

You should generate **5 (or 6) figures** and **1 table** for the programming tasks.

### Deliverables:
- A PDF file named `hw4.pdf` containing all figures and the table.
- Code files for each task:
  - `hw4_task1.py` for Figure 1
  - `hw4_task2.py` for Figure 2
  - `hw4_task3.py` for Table 1
  - `hw4_task4.py` for Figure 3
  - `hw4_task5.py` for Figure 4 and Figure 5
  - `hw4_task5b.py` for Figure 6 (if applicable)

You can generate the PDF using any tool, but a LaTeX template `hw4_Latex.txt` is provided for your convenience.
```