import tensorflow as tf
tf.random.set_seed(10)
import numpy as np
np.random.seed(10)
from scipy.optimize import minimize

class surrogate_optimizer():
    def __init__(self,trained_model,num_pars,cons):
        self.model = trained_model
        self.num_pars = num_pars
        self.cons = cons
        self.callback_array = np.zeros(shape=(1,self.num_pars),dtype='float32')

    def jac_method(self,input_var):
        input_var = input_var.reshape(1,self.num_pars).astype('float32')
        input_var = tf.Variable(input_var)
            
        with tf.GradientTape(persistent=True) as t:
            t.watch(input_var)
            # This is the objective function (square of the first variable out)
            pred = self.model(input_var)[0][0]**2
            
        return t.gradient(pred, input_var).numpy()[0,:]

    def residual(self,input_var):
        input_var = input_var.reshape(1,self.num_pars)       
        pred = self.model(input_var)[0][0]
        return ((pred)**2).numpy()

    def callbackF(self,Xi):
        sol_array = np.copy(Xi.reshape(1,self.num_pars))
        self.callback_array = np.concatenate((self.callback_array,sol_array),axis=0)

    def single_optimize(self,init_guess):
        self.solution = minimize(self.residual,init_guess,
                            jac=self.jac_method,method='BFGS',
                            tol=1e-6,options={'disp': True}, 
                            callback=self.callbackF,constraints=self.cons)

    def optimize(self,num_restarts):
        # Multiple restarts
        best_func_val = 100.0
        best_opt = None
        best_optimizer = None
        for start in range(num_restarts):
            self.init_guess = np.random.uniform(size=(1,self.num_pars))
            self.single_optimize(self.init_guess)

            # Print solution
            if self.solution.fun < best_func_val:
                best_func_val = self.solution.fun
                best_opt = np.copy(self.callback_array)
                best_optimizer = self.solution

        # Print best optimizer stats
        print('Successful? ',best_optimizer.success)
        print('Minimum function value: ',best_func_val)
        print('Parameters: ',best_optimizer.x)

        return best_func_val, best_optimizer.x, best_opt

if __name__ == '__main__':
    pass