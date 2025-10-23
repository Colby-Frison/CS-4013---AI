# hw2_genetic.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

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

def fitness_function(w, X, y):
    """Calculate fitness as e^(-er(w))"""
    error = error_function(w, X, y)
    return np.exp(-error)

def initialize_population(pop_size=20):
    """Initialize population with random weight vectors"""
    population = []
    for _ in range(pop_size):
        individual = np.array([random.choice([-1, 1]) for _ in range(6)])
        population.append(individual)
    return population

def selection(population, fitness_values):
    """Select parents based on fitness-proportional probabilities"""
    total_fitness = sum(fitness_values)
    probabilities = [f/total_fitness for f in fitness_values]
    
    # select two parents using roulette wheel selection
    parents = random.choices(population, weights=probabilities, k=2)
    return parents[0], parents[1]

def crossover(parent1, parent2):
    """Single-point crossover at the middle"""
    crossover_point = 3  # middle point
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

def mutation(individual, mutation_rate=0.05):
    """Apply mutation by flipping bits with given probability"""
    mutated = individual.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = -mutated[i]
    return mutated

def genetic_algorithm(X, y, pop_size=8, generations=50, mutation_rate=0.05):
    """Implement genetic algorithm"""
    # initialize population
    population = initialize_population(pop_size)
    best_errors = []
    
    for generation in range(generations):
        # evaluate fitness for each individual
        fitness_values = [fitness_function(ind, X, y) for ind in population]
        errors = [error_function(ind, X, y) for ind in population]
        
        # track best error
        best_error = min(errors)
        best_errors.append(best_error)
        
        # create new population
        new_population = []
        
        # elitism: keep the best individual
        best_index = np.argmin(errors)
        new_population.append(population[best_index])
        
        # generate remaining individuals
        while len(new_population) < pop_size:
            # selection
            parent1, parent2 = selection(population, fitness_values)
            
            # crossover
            child1, child2 = crossover(parent1, parent2)
            
            # mutation
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        # ensure population size doesn't exceed limit
        population = new_population[:pop_size]
    
    # find best solution in final population
    final_errors = [error_function(ind, X, y) for ind in population]
    best_index = np.argmin(final_errors)
    best_w = population[best_index]
    best_error = final_errors[best_index]
    
    return best_w, best_error, best_errors

def plot_error_curve(errors, filename='genetic_algorithm_error.png'):
    """Plot error vs generations"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(errors)), errors, 'r-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Error er(w)')
    plt.title('Genetic Algorithm: Error vs Generation')
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
    
    # run genetic algorithm
    optimal_w, optimal_error, errors = genetic_algorithm(X, y, pop_size=20, generations=50, mutation_rate=0.1)
    
    # plot results
    plot_error_curve(errors)
    
    # print results
    print("Optimal w found by genetic algorithm:", optimal_w)
    print("Optimal error er(w):", optimal_error)
    
    return optimal_w, optimal_error, errors

if __name__ == "__main__":
    optimal_w_genetic, optimal_error_genetic, errors_genetic = main()