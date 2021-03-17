#!/bin/bash
# Build custom kernels.
TF_CFLAGS=( $(python3.6 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3.6 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# Linux (pip)
g++ -std=c++11 -shared coref_kernels.cc -o coref_kernels.so -fPIC -I /usr/local/lib/python3.6/dist-packages/tensorflow/include ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} #-O2 -D_GLIBCXX_USE_CXX11_ABI=0
