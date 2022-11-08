#!/bin/python3
import tensorflow as tf

if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    print(tf.config.list_logical_devices('GPU'))
    assert(tf.config.list_physical_devices('GPU') > 0)
    assert(tf.config.list_logical_devices('GPU') > 0)
