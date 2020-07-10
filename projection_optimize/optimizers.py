import tensorflow as tf
tf.random.set_seed(10)
import numpy as np
np.random.seed(10)
from scipy.optimize import minimize
from constraints import t_lower, t_upper

temp_lift = 1.1
# For lift constraint - constrained at 1.0
def lift_eq_cons(input_var):
    global temp_lift
    return temp_lift-0.8

class surrogate_optimizer():
    def __init__(self,trained_model,num_pars,cons,lift_cons=False):
        self.model = trained_model
        self.num_pars = num_pars
        self.cons = cons
        self.callback_array = np.zeros(shape=(1,self.num_pars),dtype='float32')
        self.lift_cons_append = lift_cons

        # self.scaler_mean = tf.convert_to_tensor(trained_model.op_scaler.mean_,dtype='float32') # Required for rescaling within TF
        # self.scaler_var = tf.convert_to_tensor(trained_model.op_scaler.var_,dtype='float32') # Required for rescaling within TF

        self.scaler_min_n = trained_model.op_scaler.data_min_
        self.scaler_max_n = trained_model.op_scaler.data_max_

        self.scaler_min = tf.convert_to_tensor(trained_model.op_scaler.data_min_,dtype='float32') # Required for rescaling within TF
        self.scaler_max = tf.convert_to_tensor(trained_model.op_scaler.data_max_,dtype='float32')

    def jac_method(self,input_var):
        input_var = input_var.reshape(1,self.num_pars).astype('float32')
        input_var = tf.Variable(input_var)
            
        with tf.GradientTape(persistent=True) as t:
            t.watch(input_var)

            op = self.model(input_var)
            op = op*(self.scaler_max-self.scaler_min) + self.scaler_min
            # op = op*(self.scaler_var)+self.scaler_mean

            # This is the objective function (square of the first variable out)
            pred = 0.5*(op[0][0])**2
            
        return t.gradient(pred, input_var).numpy()[0,:].astype('double')

    def residual(self,input_var):
        # [0][0] - Drag, [0][1] - Lift
        input_var = input_var.reshape(1,self.num_pars)
        # need to rescale output
        output = self.model.op_scaler.inverse_transform(self.model.predict(input_var))
        pred = 0.5*(output[0,0])**2
        
        return pred

    def callbackF(self,Xi):
        sol_array = np.copy(Xi.reshape(1,self.num_pars))
        self.callback_array = np.concatenate((self.callback_array,sol_array),axis=0)
        
        output = self.model.op_scaler.inverse_transform(self.model.predict(Xi.reshape(1,self.num_pars)))
        # For equality constraint
        global temp_lift
        temp_lift = output[0,1]

    def single_optimize(self,init_guess):
        if self.lift_cons_append:
            self.cons.append({'type': 'eq', 'fun': lift_eq_cons})
            self.lift_cons_append = False

        self.solution = minimize(self.residual,init_guess,
                            jac=self.jac_method,method='SLSQP',
                            tol=1e-8,options={'disp': True, 'maxiter': 300, 'eps': 1.4901161193847656e-8}, 
                            callback=self.callbackF,constraints=self.cons)

    def optimize(self,num_restarts):
        # Multiple restarts
        best_func_val = 100.0
        best_opt = None
        best_optimizer = None

        for start in range(num_restarts):
            
            self.init_guess = np.random.uniform(low=np.asarray(t_lower),\
                                                high=np.asarray(t_upper))
            
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
        op = self.model(best_opt[-1].reshape(1,np.shape(self.init_guess)[0])).numpy()
        op = self.model.op_scaler.inverse_transform(op)

        print('Drag coefficient',op[0,0])
        print('Lift coefficient',op[0,1])

        return best_func_val, best_optimizer.x, best_opt

if __name__ == '__main__':
    pass