import torch

class LinearModel:

    def __init__(self):
        self.w = None 

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
        # your computation here: compute the vector of scores s
        #print("X: " +str(X))
        #print("selfw: " +str(self.w))
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
        return torch.where(scores > 0, 1.0, 0.0)

class Perceptron(LinearModel):
 
    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has 
        values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- 
        otherwise none of the formulas will work right! An easy to to make this conversion is: 
        
        y_ = 2*y - 1
        """
        
        #Accuracy of a two-label perceptron with weight w: A(w) = 1/n * sum of BINARY(s_i, y_i)
        #where s_i is the score and y_i is the target
        y_mod = 2*y - 1
        return 1 - (1.0*((LinearModel.score(self, X)*y_mod) > 0)).mean()

    def grad(self, X, y, ix = None):
        if (ix == None):
            scores = LinearModel.score(self, X) 
            
            siyi = torch.where(scores*y<0,1.0,0.0)

            return y.float()*(siyi@X)
        else:
            scores = LinearModel.score(self, X[ix,:])
            siyi = torch.where(scores*y[ix]<0,1.0,0.0)

            return torch.sum(siyi.reshape(-1,1)*(y[ix].float()@X[ix,:]),0) / len(y[ix])
        
        
class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model
    
    def step(self, X, y, lr=1.0, ix = None): 
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y.
        """
        if (ix == None):
            self.model.loss(X,y)
            self.model.w += self.model.grad(X,y)
        else:
            self.model.loss(X[ix,:],y[ix])
            self.model.w += (lr*self.model.grad(X,y,ix))