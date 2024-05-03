import torch

class LinearModel:

    def __init__(self):
        self.w = None
        self.prev_w = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))
        return X @ self.w
    
    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        #Find the score
        scores = self.score(X)
        #Use threshold to set 0 or 1 for each vector
        #Return the prediction vectors
        #Applying arbitrary threshold 0.5
        return torch.where(scores > 0.5, 1.0, 0.0)

class LogisticRegression(LinearModel):
    def __init__(self):
        super().__init__()
        
    def loss(self, X, y):
        """
        Compute the empirical risk L(w) 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector. y.size() == (n,),
            where n is the number of data points.

        RETURNS: 
            L(w), float: the empirical risk given current set of weights w
        """
        #Calculate scores and apply sigmoid function to get predictions
        #Score calculation is where weights matter since score is just X @ w
        sigmoid_score = torch.sigmoid(self.score(X))
        #Calculate loss by effectively taking an average
        return torch.sum(-y*torch.log(sigmoid_score) - (1-y)*torch.log(1 - sigmoid_score)) / X.size()[0]
    
    def grad(self, X, y):
        """
        Compute the gradient of the empirical risk del-L(w) 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector. y.size() == (n,),
            where n is the number of data points.

        RETURNS: 
            del-L(w), torch.Tensor: gradient of empirical risk. del-L(w).size() = (p,),
            where p is the number of features
        """

        sigmoid_score = torch.sigmoid(self.score(X))
        sigmoid_y = (sigmoid_score - y)

        #We turn our sigmoid_y from a (n,) tensor to a (n,1) tensor. Then we multiply each
        #sigmoid_y_i and x_i. We sum up those products across the 0th dimension, i.e., at the
        #observation level. Then we divide that sum by the number of observations, getting us
        #what is effectively an average across all observations
        return (sigmoid_y[:, None] * X).sum(dim=0) / X.size(0)
    
    def hessian(self, X, y):
        #First we find the diagonal matrix D
        #Calculated as d_kk(w) = sigmoid(s_k)(1-sigmoid(s_k))

        #Then we apply the equation X^T * D(w) X, where X^T is just the tranpose of X

        pass #TO DO


class NewtonOptimizer:
    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha):
        pass #TO DO
        
class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model

    def step(self, X, y, alpha, beta):
        """
        Complete one update of weights

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector. y.size() == (n,),
            where n is the number of data points.

            alpha, float: a hyperparameter that controls the learning rate,
            i.e., the size of change each iteration

            beta, float: a hyperparameter that controls the momentum rate.
            This allows the program to build up velocity in terms of change,
            potentially allowing faster convergence

        RETURNS: 
            None. Method updates weights but returns nothing
        """
        
        grad = self.model.grad(X,y)

        #If there is no previous weight, we exclude the previous weight from the equation
        #We apply the equation from the instructions here, which incorporates momentum
        if self.model.prev_w != None: 
            new_w = self.model.w - alpha * grad + beta * (self.model.w - self.model.prev_w)
        else:
            new_w = self.model.w - alpha * grad + beta * (self.model.w)
        self.model.prev_w = self.model.w
        self.model.w = new_w