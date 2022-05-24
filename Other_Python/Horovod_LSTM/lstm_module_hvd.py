import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt

#(1)
import horovod.tensorflow as hvd
hvd.init()
print("Horovod: I am rank %d of %d"%(hvd.rank(), hvd.size()))

#(2)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimentahl.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


#Build the model which does basic map of inputs to coefficients
class standard_lstm(Model):
    def __init__(self,data):
        super(standard_lstm, self).__init__()
        # Set up the data for the LSTM
        self.data_tsteps = np.shape(data)[0]
        self.state_len = np.shape(data)[1]

        self.preproc_pipeline = Pipeline([('stdscaler', StandardScaler()),('minmax', MinMaxScaler(feature_range=(-1, 1)))])
        self.data = self.preproc_pipeline.fit_transform(data)
        # Need to make minibatches
        self.seq_num = 5
        self.total_size = np.shape(data)[0]-int(self.seq_num) # Limit of sampling

        input_seq = np.zeros(shape=(self.total_size,self.seq_num,self.state_len))  #[samples,n_inputs,state_len]
        output_seq = np.zeros(shape=(self.total_size,self.state_len)) #[samples,n_outputs,state_len]

        snum = 0
        for t in range(0,self.total_size):
            input_seq[snum,:,:] = self.data[None,t:t+self.seq_num,:]
            output_seq[snum,:] = self.data[None,t+self.seq_num,:]        
            snum = snum + 1

        # Shuffle dataset
        idx = np.arange(snum)
        np.random.shuffle(idx)
        input_seq = input_seq[idx]
        output_seq = output_seq[idx]
        # Split into train and test
        self.input_seq_test = input_seq[int(0.9*snum):]
        self.output_seq_test = output_seq[int(0.9*snum):]
        input_seq = input_seq[:int(0.9*snum)]
        output_seq = output_seq[:int(0.9*snum)]

        # Split into train and valid
        self.ntrain = int(0.8*np.shape(input_seq)[0])
        self.nvalid = np.shape(input_seq)[0] - self.ntrain

        self.input_seq_train = input_seq[:self.ntrain]
        self.output_seq_train = output_seq[:self.ntrain]

        self.input_seq_valid = input_seq[self.ntrain:]
        self.output_seq_valid = output_seq[self.ntrain:]

        #(7) Horovod: shard
        input_seq_test_shard = self.input_seq_test[hvd.rank()::hvd.size()]
        self.input_seq_test = input_seq_test_shard 
        output_seq_test_shard = self.output_seq_test[hvd.rank()::hvd.size()]
        self.output_seq_test = output_seq_test_shard
        input_seq_train_shard = self.input_seq_train[hvd.rank()::hvd.size()]
        self.input_seq_train = input_seq_train_shard
        output_seq_train_shard = self.output_seq_train[hvd.rank()::hvd.size()]
        self.output_seq_train = output_seq_train_shard
        input_seq_valid_shard = self.input_seq_valid[hvd.rank()::hvd.size()]
        self.input_seq_valid = input_seq_valid_shard
        output_seq_valid_shard = self.output_seq_valid[hvd.rank()::hvd.size()]
        self.output_seq_valid = output_seq_valid_shard

        # Define architecture
        xavier=tf.keras.initializers.GlorotUniform()

        self.l1=tf.keras.layers.LSTM(50,return_sequences=True,input_shape=(self.seq_num,self.state_len))
        self.l2=tf.keras.layers.LSTM(50,return_sequences=False)
        self.out = tf.keras.layers.Dense(self.state_len)
        self.model = tf.keras.Sequential([self.l1, self.l2, self.out])
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001*hvd.size()) #(3)

    
    # Running the model
    def call(self,X, train=True):
        '''
        h1 = self.l1(X)
        h2 = self.l2(h1)
        out = self.out(h2)
        '''
        out = self.model(X, training=train)
        return out
    
    # Regular MSE
    def get_loss(self,X,Y):
        op=self.call(X)
        loss = tf.reduce_mean(tf.math.square(op-Y))
        return loss

    # get gradients - regular
    def get_grad(self,X,Y):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(X,Y)
        #(4)
        tape = hvd.DistributedGradientTape(tape)
        g = tape.gradient(L, self.trainable_variables)

        return g
    
    # perform gradient descent - regular
    def network_learn(self,X,Y):
        g = self.get_grad(X,Y)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Train the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 10
        best_valid_loss = 999999.0 # Some large number 

        self.num_batches = 20
        self.train_batch_size = int(self.ntrain/self.num_batches)
        self.valid_batch_size = int(self.nvalid/self.num_batches)
        #self.train_batch_size = 64
        #self.valid_batch_size = 16 
        #self.num_batches = int(self.ntrain/self.train_batch_size/hvd.size())
        #self.num_batches_valid = int((self.nvalid)/self.valid_batch_size/hvd.size())
        
        for i in range(10):
            # (5)
            if (i==0 and batch==0):
                hvd.broadcast_variables(self.model.variables, root_rank=0)
                hvd.broadcast_variables(self.train_op.variables(), root_rank=0)

            # Training loss
            print('Training iteration:',i)
            for batch in range(self.num_batches//hvd.size()):
                input_batch = self.input_seq_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                output_batch = self.output_seq_train[batch*self.train_batch_size:(batch+1)*self.train_batch_size]
                self.network_learn(input_batch,output_batch)
            # Validation loss
            valid_loss = 0.0
            valid_r2 = 0.0

            for batch in range(self.num_batches//hvd.size()):
                input_batch = self.input_seq_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]
                output_batch = self.output_seq_valid[batch*self.valid_batch_size:(batch+1)*self.valid_batch_size]

                valid_loss = valid_loss + self.get_loss(input_batch,output_batch).numpy()
                predictions = self.call(self.input_seq_valid)
                # order important: r2_score(y_true, y_pred)
                valid_r2 = valid_r2 + r2_score(self.output_seq_valid, predictions)

            total_valid_loss = hvd.allreduce(valid_loss, average=True)
            valid_loss = total_valid_loss
            
            valid_r2 = valid_r2/(batch+1)


            # Check early stopping criteria
            if valid_loss < best_valid_loss:
                if (hvd.rank()==0):
                    print('Improved validation loss from:',best_valid_loss,' to:', valid_loss)
                    print('Validation R2:',valid_r2)
                
                best_valid_loss = valid_loss

                #(6)
                if hvd.rank() == 0:
                    self.save_weights('./checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                if (hvd.rank()==0):
                    print('Validation loss (no improvement):',valid_loss)
                    print('Validation R2:',valid_r2)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break
                
        # Check accuracy on test
        predictions = self.call(self.input_seq_test, train=False)
        loss = self.get_loss(self.input_seq_test,self.output_seq_test).numpy()
        total_loss = hvd.allreduce(loss, average=True)
        
        if (hvd.rank()==0):
            print('Test loss:',self.get_loss(self.input_seq_test,self.output_seq_test).numpy())
        r2 = r2_score(self.output_seq_test, predictions)
        print('Test R2:',r2)
        r2_iter = 0

    # Load weights
    def restore_model(self):
        #(6)
        if hvd.rank() == 0:
            self.load_weights('./checkpoints/my_checkpoint') # Load pretrained model
            print('Model restored successfully!')

    # Do some testing
    def model_inference(self,test_data):
        # Restore from checkpoint
        self.restore_model()

        if hvd.rank() == 0:

            # Scale testing data
            #test_data = self.preproc_pipeline.fit_transform(test_data, callbacks=callbacks) #(5)
            test_data = self.preproc_pipeline.fit_transform(test_data)

            # Test sizes
            test_total_size = np.shape(test_data)[0]-int(self.seq_num) # Limit of sampling

            # Input placeholder setup
            rec_input_seq = test_data[:self.seq_num,:].reshape(1,self.seq_num,self.state_len)
            
            # True outputs
            rec_output_seq = np.zeros(shape=(test_total_size,self.state_len)) #[samples,n_outputs,state_len]
            snum = 0
            for t in range(test_total_size):
                rec_output_seq[snum,:] = test_data[None,t+self.seq_num,:]        
                snum = snum + 1

            # Recursive predict
            print('Making predictions on testing data')
            rec_pred = np.copy(rec_output_seq)
            for t in range(test_total_size):
                rec_pred[t] = self.call(rec_input_seq).numpy()[0]
                rec_input_seq[0,0:-1,:] = rec_input_seq[0,1:]
                rec_input_seq[0,-1,:] = rec_pred[t]

            # Rescale
            rec_pred = self.preproc_pipeline.inverse_transform(rec_pred)
            rec_output_seq = self.preproc_pipeline.inverse_transform(rec_output_seq)

            # plot
            for i in range(self.state_len):
                plt.figure()
                plt.title('Mode '+str(i))
                plt.plot(rec_pred[:,i],label='Predicted')
                plt.plot(rec_output_seq[:,i],label='True')
                plt.legend()
                plt.savefig('Mode_'+str(i)+'_prediction.png')
                plt.close()

            return rec_output_seq, rec_pred

if __name__ == '__main__':
    print('Architecture file')

