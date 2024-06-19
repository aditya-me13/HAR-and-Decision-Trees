import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from latex import latexify, format_axes

from sklearn.tree import DecisionTreeClassifier

latexify()

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# N = 30
# P = 5
# X = pd.DataFrame(np.random.randn(N, P))
# y = pd.Series(np.random.randn(N))

# Function to create fake data (take inspiration from usage.py)
def give_data(category, N, P):
    # category:
    # 1 -> RIRO
    # 2 -> RIDO
    # 3 -> DIDO
    # 4 -> DIRO
    
    if (category == 1):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
        
    elif (category == 2):
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(P, size = N), dtype = "category")
        
    elif (category == 3):
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size = N), dtype = "category") for i in range(5)})
        y = pd.Series(np.random.randint(P, size = N), dtype = "category")
        
    elif (category == 4):
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size = N), dtype = "category") for i in range(5)})
        y = pd.Series(np.random.randn(N))
    
    
    return X, y
# ...
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def time_data(manyX, manyY, model):
    TimeFit, TimePredict, Shapes = [], [], []
    for X, y in zip(manyX, manyY):
        t1 = time.time()
        model.fit(X, y)
        t2 = time.time()
        T1 = t2 - t1
        
        t3 = time.time()
        y_hat = model.predict(X)
        t4 = time.time()
        T2 = t4 - t3
        
        M, N = X.shape[0], X.shape[1]
        TimeFit.append(T1)
        TimePredict.append(T2)
        Shapes.append(f"(M: {M}, N: {N})")


    
    return np.array(TimeFit), np.array(TimePredict), np.array(Shapes)
    
    
    
# ...
# Function to plot the results
def plot_result(TimeFit, TimePredict):
    plt.figure(figsize = (10, 6))
    plt.plot(range(len(TimeFit)), TimeFit, color = "r", marker = "o")
    plt.xlabel("Model Number")
    plt.ylabel("Fitting Time (s)")
    plt.title("Fitting Times for different models")
    #plt.text(, 0.8, f"Average : {np.mean(TimeFit)}\nStd: {np.std(TimeFit)}", bbox={"facecolor" : "blue", "alpha" : 0.5, "pad": 10})
    plt.grid()
    plt.show()
    
    plt.figure(figsize = (10, 6))
    plt.plot(range(len(TimePredict)), TimePredict, color = "g", marker = "s")
    plt.xlabel("Model Number")
    plt.ylabel("Predicting Time (s)")
    plt.title("Predicting Times for different models")
    #plt.text(0.8, 0.8, f"Average : {np.mean(TimePredict)}\nStd: {np.std(TimePredict)}", bbox={"facecolor" : "green", "alpha" : 0.5, "pad": 10})
    plt.grid()
    plt.show()



# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots


# category:
# 1 -> RIRO
# 2 -> RIDO
# 3 -> DIDO
# 4 -> DIRO

Xs, ys = [], []
for i in range(num_average_time):
    N, P = np.random.randint(low = 20, high = 50), np.random.randint(low = 3, high = 15)
    X, y = give_data(2, N, P)
    Xs.append(X)
    ys.append(y)

plot_result(time_data(Xs, ys, DecisionTreeClassifier())[0], time_data(Xs, ys, DecisionTreeClassifier())[1])
# print(Xs[0])
    