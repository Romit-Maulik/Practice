gcc -I/home/rmlans/Desktop/Tutorials/TF_C_API/include -L/home/rmlans/Desktop/Tutorials/TF_C_API/lib hello_tf.c -ltensorflow -o hello_tf
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rmlans/Desktop/Tutorials/TF_C_API/lib
./hello_tf