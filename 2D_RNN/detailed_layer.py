import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(dir_path)

import tensorflow as tf
tf.random.set_seed(10)
tf.keras.backend.set_floatx('float32')


# Special layer to have more control over LSTM encoder cell
class LSTM_encoder_layer(tf.keras.layers.Layer):
    def __init__(self, input_dim_list, seq_len_list):
        super(LSTM_encoder_layer, self).__init__()
        self.num_dof = len(input_dim_list)
        self.input_dim_list = input_dim_list
        self.seq_len_list = seq_len_list

        self.initialize_layer()

    def initialize_layer(self):

        var_init = tf.random_normal_initializer()
        
        self.wu_list = []
        self.wf_list = []
        self.wo_list = []
        self.wc_list = []

        self.bu_list = []
        self.bf_list = []
        self.bo_list = []
        self.bc_list = []

        self.m_list = []
        self.h_list = []

        for i in range(self.num_dof):
            self.wu_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(2*self.input_dim_list[i], self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.wf_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(2*self.input_dim_list[i], self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.wo_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(2*self.input_dim_list[i], self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.wc_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(2*self.input_dim_list[i], self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )


            self.bu_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.bf_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.bo_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.bc_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

    @tf.function
    def call(self, input_list):

        hh_list = []
        mm_list = []


        for j in range(self.num_dof): # For each state vector input
       
            seq_length = self.seq_len_list[j]

            hh = tf.zeros(shape=(tf.shape(input_list[j])[0],self.input_dim_list[j]),dtype='float32') # batch_size x state_dimension
            mm = tf.zeros(shape=(tf.shape(input_list[j])[0],self.input_dim_list[j]),dtype='float32') # batch_size x state_dimension
            
            # For time dimension unroll
            for i in range(seq_length):
                raw_inputs = input_list[j][:,i]
                inputs = tf.concat([raw_inputs,hh],axis=-1)

                gu = tf.nn.sigmoid(tf.matmul(inputs,self.wu_list[j]) + self.bu_list[j])
                gf = tf.nn.sigmoid(tf.matmul(inputs,self.wf_list[j]) + self.bf_list[j])
                go = tf.nn.sigmoid(tf.matmul(inputs,self.wo_list[j]) + self.bo_list[j])
                gc = tf.nn.tanh(tf.matmul(inputs,self.wc_list[j]) + self.bc_list[j])

                mm = gf*mm + gu*gc
                hh = tf.nn.tanh(go*mm)

            hh_list.append(hh)
            mm_list.append(mm)

        return hh_list, mm_list


# Special layer to have more control over LSTM grid cell
class LSTM_grid_layer(tf.keras.layers.Layer):
    def __init__(self, grid_dim, output_dim_list, seq_len_list):
        super(LSTM_grid_layer, self).__init__()
        self.num_dof = len(output_dim_list)
        self.output_dim_list = output_dim_list
        self.seq_len_list = seq_len_list
        self.grid_dim = grid_dim

        self.initialize_layer()

    def initialize_layer(self):

        var_init = tf.random_normal_initializer()
        
        self.wu_list = []
        self.wf_list = []
        self.wo_list = []
        self.wc_list = []

        self.bu_list = []
        self.bf_list = []
        self.bo_list = []
        self.bc_list = []

        self.m_list = []
        self.h_list = []

        for i in range(self.num_dof):
            self.wu_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.grid_dim, self.output_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.wf_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.grid_dim, self.output_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.wo_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.grid_dim, self.output_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.wc_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.grid_dim, self.output_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )


            self.bu_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.output_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.bf_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.output_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.bo_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.output_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.bc_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.output_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

    @tf.function
    def call(self, hidden, memory):
        '''
        Hidden and memory are lists of tensors for the hidden state and the memory of different inputs
        '''

        for j in range(self.num_dof): # For each state vector
       
            seq_length = self.seq_len_list[j]

            # For time dimension unroll
            for i in range(seq_length):
                inputs = tf.concat([hidden[j],memory[j]],axis=-1)

                gu = tf.nn.sigmoid(tf.matmul(inputs,self.wu_list[j]) + self.bu_list[j])
                gf = tf.nn.sigmoid(tf.matmul(inputs,self.wf_list[j]) + self.bf_list[j])
                go = tf.nn.sigmoid(tf.matmul(inputs,self.wo_list[j]) + self.bo_list[j])
                gc = tf.nn.tanh(tf.matmul(inputs,self.wc_list[j]) + self.bc_list[j])

                memory[j] = gf*memory[j] + gu*gc
                hidden[j] = tf.nn.tanh(go*memory[j])


        return hidden, memory


# Special layer to have more control over LSTM grid cell
class Original_LSTM_grid_layer(tf.keras.layers.Layer):
    def __init__(self, input_dim_list, seq_length):
        super(Original_LSTM_grid_layer, self).__init__()
        self.num_dof = len(input_dim_list)
        self.input_dim_list = input_dim_list
        self.seq_length = seq_length
        self.state_len = sum(input_dim_list)

        self.initialize_layer()

    def initialize_layer(self):

        var_init = tf.random_normal_initializer()
        
        self.wu_list = []
        self.wf_list = []
        self.wo_list = []
        self.wc_list = []

        self.bu_list = []
        self.bf_list = []
        self.bo_list = []
        self.bc_list = []

        for i in range(self.num_dof):
            self.wu_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(2*self.input_dim_list[i], self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.wf_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(2*self.input_dim_list[i], self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.wo_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(2*self.input_dim_list[i], self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.wc_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(2*self.input_dim_list[i], self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )


            self.bu_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.bf_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.bo_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.bc_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )


            self.grid_wu_list = []
            self.grid_wf_list = []
            self.grid_wo_list = []
            self.grid_wc_list = []

            self.grid_bu_list = []
            self.grid_bf_list = []
            self.grid_bo_list = []
            self.grid_bc_list = []


        for i in range(self.num_dof):
            self.grid_wu_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.state_len, self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.grid_wf_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.state_len, self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.grid_wo_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.state_len, self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )

            self.grid_wc_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.state_len, self.input_dim_list[i]), dtype="float32"),
                            trainable=True
                            )
                )


            self.grid_bu_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.grid_bf_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.grid_bo_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )

            self.grid_bc_list.append(
                tf.Variable(
                            initial_value=var_init(shape=(self.input_dim_list[i],), dtype="float32"),
                            trainable=True
                            )
                )


    # @tf.function
    def call(self, inputs_list):

        h_list = []
        m_list = []
        for i in range(self.num_dof):
            batch_dim = tf.shape(inputs_list[i])[0]
            state_dim = tf.shape(inputs_list[i])[2]

            hidden = tf.zeros(shape=(batch_dim,state_dim),dtype='float32')
            memory = tf.zeros(shape=(batch_dim,state_dim),dtype='float32')
        
            h_list.append(hidden)
            m_list.append(memory)

        # For time dimension unroll
        for i in range(self.seq_length):
            # For each grid input
            for j in range(self.num_dof):

                temp_input_dim = tf.convert_to_tensor(inputs_list[j])[:,i]
                inputs = tf.concat([temp_input_dim,h_list[j]],axis=-1)

                gu = tf.nn.sigmoid(tf.matmul(inputs,self.wu_list[j]) + self.bu_list[j])
                gf = tf.nn.sigmoid(tf.matmul(inputs,self.wf_list[j]) + self.bf_list[j])
                go = tf.nn.sigmoid(tf.matmul(inputs,self.wo_list[j]) + self.bo_list[j])
                gc = tf.nn.tanh(tf.matmul(inputs,self.wc_list[j]) + self.bc_list[j])

                m_list[j] = gf*m_list[j] + gu*gc
                h_list[j] = tf.nn.tanh(go*m_list[j])


            hgrid = tf.concat(h_list,axis=-1)

            # For each grid input
            for j in range(self.num_dof):

                gu = tf.nn.sigmoid(tf.matmul(hgrid,self.grid_wu_list[j]) + self.grid_bu_list[j])
                gf = tf.nn.sigmoid(tf.matmul(hgrid,self.grid_wf_list[j]) + self.grid_bf_list[j])
                go = tf.nn.sigmoid(tf.matmul(hgrid,self.grid_wo_list[j]) + self.grid_bo_list[j])
                gc = tf.nn.tanh(tf.matmul(hgrid,self.grid_wc_list[j]) + self.grid_bc_list[j])

                m_list[j] = gf*m_list[j] + gu*gc
                h_list[j] = tf.nn.tanh(go*m_list[j])

        return h_list, m_list