According to the plots in the experiment.ipynb

1) The order for prediction time taken by DIDO, DIRO, RIRO, RIDO when the sklearn decision tree is used:

DIDO< DIRO< RIDO< RIRO

2) The order for fitting time taken by DIDO, DIRO, RIRO, RIDO is when decision tree built by us is used:

RIDO< DIRO< DIDO< RIRO

3) The order for prediction time taken by DIDO, DIRO, RIRO, RIDO is when decision tree built by us is used:

DIRO< DIDO< RIRO< RIDO

* Our decision tree model uses the least time for Real Input Discrete Output case because of the fact that the tree obtained in case of real input is binary, i.e., the split at every level is binary, and thus uses lesser time to fit than the discrete input one's. 

* The Real Input Real output tree takes the most time because to get the optimal attribute we need to check mse for every two consecutive pairwise values of all the attributes.

* Real-input decision trees may need to grow deeper to capture complex relationships in the data, leading to more nodes and branches in the tree. This increased depth can contribute to longer prediction times.

* In all the cases time taken by our decision tree is almost 100 times the time taken by sklearn's model because we have used multiple pointers and loops to check for different corner cases, which in our case increases the complexity.


