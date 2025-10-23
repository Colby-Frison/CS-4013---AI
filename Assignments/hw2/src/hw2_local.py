# hw2_local.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data(filename='CreditCard.csv'):
    """Load and preprocess the credit card data"""
    data = pd.read_csv(filename)
    
    # encode categorical variables
    gender_map = {'M': 1, 'F': 0}
    data['Gender'] = data['Gender'].map(gender_map)
    
    owner_map = {'Y': 1, 'N': 0}
    data['CarOwner'] = data['CarOwner'].map(owner_map)
    data['PropertyOwner'] = data['PropertyOwner'].map(owner_map)
    
    # handle missing values
    data['Gender'] = data['Gender'].fillna(0)
    data['CarOwner'] = data['CarOwner'].fillna(0)
    data['PropertyOwner'] = data['PropertyOwner'].fillna(0)
    data['#Children'] = data['#Children'].fillna(0)
    data['WorkPhone'] = data['WorkPhone'].fillna(0)
    data['Email_ID'] = data['Email_ID'].fillna(0)
    
    # extract features and target
    X = data[['Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID']].values
    y = data['CreditApprove'].values
    
    return X, y

def error_function(w, X, y):
    """Calculate the error function er(w)"""
    n = len(y)
    predictions = X @ w  # matrix multiplication
    error = np.sum((predictions - y) ** 2) / n
    return error

def get_adjacent_solutions(w):
    """Generate all adjacent solutions (differ by exactly one element)"""
    adjacent = []
    for i in range(len(w)):
        # flip the i-th element
        new_w = w.copy()
        new_w[i] = -new_w[i]
        adjacent.append(new_w)
    return adjacent

def hill_climbing_search(X, y, max_rounds=100):
    """Implement hill climbing local search algorithm"""
    # initialize with starting solution
    current_w = np.array([-1, -1, -1, -1, -1, -1])
    current_error = error_function(current_w, X, y)
    
    errors = [current_error]
    rounds = 0
    
    while rounds < max_rounds:
        rounds += 1
        
        # generate all adjacent solutions
        adjacent_solutions = get_adjacent_solutions(current_w)
        
        # evaluate all adjacent solutions
        best_adjacent_error = float('inf')
        best_adjacent_w = None
        
        for adjacent_w in adjacent_solutions:
            adjacent_error = error_function(adjacent_w, X, y)
            if adjacent_error < best_adjacent_error:
                best_adjacent_error = adjacent_error
                best_adjacent_w = adjacent_w
        
        # check if we found a better solution
        if best_adjacent_error < current_error:
            current_w = best_adjacent_w
            current_error = best_adjacent_error
            errors.append(current_error)
        else:
            # no improvement, convergence reached
            break
    
    return current_w, current_error, errors

def plot_error_curve(errors, filename='local_search_error.png'):
    """Plot error vs search rounds"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(errors)), errors, 'b-', linewidth=2)
    plt.xlabel('Search Round')
    plt.ylabel('Error er(w)')
    plt.title('Hill Climbing Local Search: Error vs Search Round')
    plt.grid(True, alpha=0.3)
    
    # save figure to current directory and Figures subdirectory
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    try:
        plt.savefig(f"Figures/{filename}", dpi=300, bbox_inches='tight')
    except:
        # create Figures directory if it doesn't exist
        import os
        os.makedirs("Figures", exist_ok=True)
        plt.savefig(f"Figures/{filename}", dpi=300, bbox_inches='tight')
    
    plt.close()

def main():
    # load data
    import os
    # get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # construct the full path to the CSV file
    csv_path = os.path.join(script_dir, 'CreditCard.csv')
    X, y = load_and_preprocess_data(csv_path)
    
    # run hill climbing search
    optimal_w, optimal_error, errors = hill_climbing_search(X, y)
    
    # plot results
    plot_error_curve(errors)
    
    # print results
    print("Optimal w found by local search:", optimal_w)
    print("Optimal error er(w):", optimal_error)
    
    return optimal_w, optimal_error, errors

if __name__ == "__main__":
    optimal_w_local, optimal_error_local, errors_local = main()