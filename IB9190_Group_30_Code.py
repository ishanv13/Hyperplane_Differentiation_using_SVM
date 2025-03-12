# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:41:03 2025

@author: AAMA Group 30
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from scipy.optimize import linprog


# Load Data
file_name = 'Data.csv'
df = pd.read_csv(file_name, header=None, index_col=False)
# Create label y for Iris-setosa=1 and everything else=-1
df['label'] = df[4].apply(lambda x: 1 if x == 'Iris-setosa' else -1)

X = df.iloc[:, :4].values  # Features
y = df['label'].values  # Labels
num_samples, num_features = X.shape


# Find the initial feasible solution with LP
def lp_initialization(X, y, num_features, M):
    # Minimise 0 as objective
    c = np.zeros(num_features + 1)
    
    # Constraints: y_i * (w^T x_i + b) ≥ 1
    A = -y[:, np.newaxis] * np.hstack((X, np.ones((len(X), 1))))
    b = -np.ones(len(X))  # Reformulated to ≤ constraint for linprog
    
    # Bounds for w and b
    bounds = [(-M, M)] * num_features + [(-M, M)]
    
    # Solve the LP problem
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        return res.x[:-1], res.x[-1]  # w, b
    else:
        raise ValueError("No feasible solution found.")


# Compute feasible direction
def direction(w_k, X, y, M):
    model = pyo.ConcreteModel()
    model.j = pyo.RangeSet(len(w_k))
    model.i = pyo.RangeSet(len(X))
    model.v_w = pyo.Var(model.j, bounds=(-M, M))
    model.v_b = pyo.Var(bounds=(-M, M))
    
    model.obj = pyo.Objective(expr=sum(w_k[j - 1] * model.v_w[j] for j in model.j), sense=pyo.minimize)
    
    def svm_constraint(model, i):
        return y[i - 1] * (sum(model.v_w[j] * X[i - 1, j - 1] for j in model.j) + model.v_b) >= 1
    
    model.svm_constr = pyo.Constraint(model.i, rule=svm_constraint)
    solver = SolverFactory('glpk')
    solver.solve(model)
    
    v_w = np.array([pyo.value(model.v_w[j]) for j in model.j])
    v_b = pyo.value(model.v_b)
    return v_w, v_b

# Compute step size
def step_size(w_k, d_w):
    model = pyo.ConcreteModel()
    model.tau = pyo.Var(bounds=(0, 1))
    
    def objective_rule(model):
        w_new = w_k + model.tau * d_w
        return 0.5 * sum(w_new[j]**2 for j in range(len(w_new)))
    
    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    solver = SolverFactory('ipopt')
    solver.solve(model)
    
    return pyo.value(model.tau)

# Define constraint check function
def check_constraints(X, y, w_opt, b_opt):
    violations = sum(y[i] * (np.dot(X[i], w_opt) + b_opt) < 1 for i in range(len(X)))
    return violations == 0, violations

# Calculate accuracy
def accuracy(X, y, w, b):
    predictions = np.sign(np.dot(X, w) + b)
    accuracy = np.mean(predictions == y)
    return accuracy

# Update solution and repeat until convergence
M_values = [1, 10, 50, 100]
epsilon_values = [0.01, 1e-4, 1e-6, 1e-8]
results = []

for M in M_values:
    for epsilon in epsilon_values:
        # Use LP-based initialization
        w0, b0 = lp_initialization(X, y, num_features, M)
        
        w_k, b_k = w0, b0
        previous_w, previous_b = np.zeros(num_features), 0
        max_iterations, iteration = 500, 0  
        
        while iteration < max_iterations:
            v_w, v_b = direction(w_k, X, y, M)
            d_w, d_b = v_w - w_k, v_b - b_k
            tau_k = step_size(w_k, d_w)
            
            w_k, b_k = w_k + tau_k * d_w, b_k + tau_k * d_b
            norm_diff = np.linalg.norm(w_k - previous_w) + abs(b_k - previous_b)
            if norm_diff < epsilon:
                break
            
            previous_w, previous_b = w_k, b_k
            iteration += 1
        
        # Compute the objective function value
        objective_value = 0.5 * np.linalg.norm(w_k)**2
        
        # Check constraints
        satisfied, violations = check_constraints(X, y, w_k, b_k)
        
        # Compute accuracy score
        accuracy_score = accuracy(X, y, w_k, b_k)
        
        # Add everything to results df
        results.append([M, epsilon, objective_value, w_k.tolist(), b_k, tau_k, w0.tolist(), b0, iteration, satisfied, violations, accuracy_score])

# Store results in a DataFrame
results_df = pd.DataFrame(results, columns=['M', 'Epsilon', 'Objective Value', 'w_opt', 'b_opt', 'tau_k', 'w0', 'b0', 'Iterations', 'Constraints Satisfied', 'Violations', 'Accuracy'])
print(results_df)

# Takes around 3 minutes to run
