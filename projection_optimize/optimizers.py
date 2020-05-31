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
        self.scaler_min = tf.convert_to_tensor(trained_model.op_scaler.data_min_) # Required for rescaling within TF
        self.scaler_max = tf.convert_to_tensor(trained_model.op_scaler.data_max_)

    def jac_method(self,input_var):
        input_var = input_var.reshape(1,self.num_pars).astype('float32')
        input_var = tf.Variable(input_var)
            
        with tf.GradientTape(persistent=True) as t:
            t.watch(input_var)

            op = self.model(input_var)
            op = op*(self.scaler_max-self.scaler_min) + self.scaler_min

            # This is the objective function (square of the first variable out)
            pred = 0.5*(op[0][0]**2)
            
        return t.gradient(pred, input_var).numpy()[0,:].astype('double')

    def residual(self,input_var):
        # [0][0] - Drag, [0][1] - Lift
        input_var = input_var.reshape(1,self.num_pars)
        # need to rescale output
        output = self.model.op_scaler.inverse_transform(self.model.predict(input_var))
        pred = 0.5*(output[0,0])**2# + (self.model(input_var)[0][1]-0.3)**2
        
        return pred

    def callbackF(self,Xi):
        sol_array = np.copy(Xi.reshape(1,self.num_pars))
        self.callback_array = np.concatenate((self.callback_array,sol_array),axis=0)

    def single_optimize(self,init_guess):
        self.solution = minimize(self.residual,init_guess,
                            jac=self.jac_method,method='SLSQP',
                            tol=1e-7,options={'disp': True, 'maxiter': 20}, 
                            callback=self.callbackF,constraints=self.cons)

    def optimize(self,num_restarts):
        # Multiple restarts
        best_func_val = 100.0
        best_opt = None
        best_optimizer = None

        for start in range(num_restarts):
            
            self.init_guess = np.random.uniform(low=np.asarray([-0.08876, -0.3269, -0.40838, -0.14721, -0.08876, -0.37975, -0.35672, -0.04067]),\
                                                high=np.asarray([0.1648, 0.6071, 0.75842, 0.27339, 0.1648, 0.70525 , 0.66248, 0.07553]))
            
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

        # Print drag and lift
        op = self.model(best_opt[-1].reshape(1,8)).numpy()
        op = self.model.op_scaler.inverse_transform(op)

        print('Drag coefficient',op[0,0])
        print('Lift coefficient',op[0,1])

        return best_func_val, best_optimizer.x, best_opt

if __name__ == '__main__':
    pass