class MyLineReg():
    def __init__(self, n_iter, learning_rate, metric = None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        
    def count_metric(self, y, pred):
        if self.metric == 'mae':
            result = np.abs(y - pred).mean()
        elif self.metric == 'mse':
            result = ((y - pred) ** 2).mean()
        elif self.metric == 'rmse':
            result = ((y - pred) ** 2).mean() ** 0.5
        elif self.metric == 'mape':
            result = abs((y - pred) / y).mean() * 100
        elif self.metric == 'r2':
            result = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        
        return result
    
    def fit(self, X, y, verbose = False):
        X.insert(0, 'Intercept', 1)
        cols_quantity = X.shape[1]
        obj_quantity = X.shape[0]
        self.weights = np.ones(cols_quantity)
        loss_mse = 0
               
        y_mean = y.mean()        
        
        for i in range(self.n_iter):
            # for metrics
            squares, mape = 0.0, 0.0
            variance = 0
            
            pred = np.dot(X, self.weights)
            
            loss_mse = ((y - pred) ** 2).mean()
            grad = 2 / obj_quantity * (pred - y) @ X
            self.weights = self.weights - self.learning_rate * grad
            
            
            # count metrics
            if self.metric != None:
                pred = np.dot(X, self.weights)
                self.metric_amount = self.count_metric(y, pred)

            # вывод статистики на экран
            if verbose == True:
                if i == 0:
                    if self.metric == None:
                        print(f'start | loss: {loss_mse}')
                    else:
                        print(f'start | loss: {loss_mse} | {self.metric}: {self.metric_amount}')
                elif i % verbose == 0:
                    if self.metric == None:
                        print(f'{i} | loss: {loss_mse}')
                    else:
                        print(f'{i} | loss: {loss_mse} | {self.metric}: {self.metric_amount}')  
        
       
    def get_coef(self):
        return self.weights[1:]

    def predict(self, X):
        X.insert(0, 'Intercept', 1)
        pred = np.dot(X, self.weights)
        return pred
    
    def get_best_score(self):
        return self.metric_amount


