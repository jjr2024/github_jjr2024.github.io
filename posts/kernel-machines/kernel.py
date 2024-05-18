import torch
   
class KernelLogisticRegression:
    def __init__(self, kernel, lam = 0.1, gamma = 0.1):
        self.a = None #our weights
        self.Xt = None #saves our training data
        self.transpose_k = None #stores the tranpose of k to avoid recalculations
        self.lossvec = None #keeps track of our losses
        self.kernel = kernel
        self.lam = lam
        self.gamma = gamma

    def score(self, X, recompute_kernel = False):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.a, x[i]>. 

        If self.a currently has value None, then it is necessary to first initialize self.a to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            recompute_kernel, boolean: determines whether or not we should recompute k

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,p)
        """
        if self.a is None: 
            self.a = torch.rand((X.size()[0])) / 10
        
        if recompute_kernel == True:
            self.transpose_k = torch.t(self.kernel(X,self.Xt,self.gamma))
        
        return self.a @ self.transpose_k  #outputs n x p tensor
        
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
        l1_norm = torch.sum(torch.abs(self.a))

        return -(torch.mean(y*torch.log(sigmoid_score) + (1-y)*torch.log(1 - sigmoid_score))) + (self.lam*l1_norm)
    
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

        grad = (self.transpose_k @ sigmoid_y) / X.size(0)
        grad += self.lam * torch.sign(self.a)  # Gradient of L1 regularization
        return grad
    
    def step(self, X, y, lr):
        """
        Complete one update of weights

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector. y.size() == (n,),
            where n is the number of data points.

            lr, float: a hyperparameter that controls the learning rate,
            i.e., the size of change each iteration

        RETURNS: 
            None. Method updates weights but returns nothing
        """
        grad = self.grad(X, y)
        self.a = self.a - (lr * grad)
    
    def fit(self, X, y, m_epochs = 1000, lr = 0.001):
        """
        Fits our model to input training data

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s.

            y, torch.Tensor: the target vector. y.size() == (n,),
            where n is the number of data points.

            m_epochs, integer: the number of iterations, i.e., the
            overall number of times we call step()

            lr, float: a hyperparameter that controls the learning rate,
            i.e., the size of change each iteration

        RETURNS: 
            None. Method updates weights and adds to the loss vector but returns nothing
        """
        #Save training data
        self.Xt = X
        loss_vec = []

        #Calculate the tranpose of k here to avoid doing the calculation more than once
        self.transpose_k = torch.t(self.kernel(X,self.Xt,self.gamma))

        for _ in range(m_epochs):
            loss = self.loss(X, y) 
            loss_vec.append(loss)
            self.step(X, y, lr)
        self.lossvec = loss_vec