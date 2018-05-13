import tensorflow as tf
import numpy as np
import csv
import pdb
from numpy import genfromtxt

def read_data():
    my_data = genfromtxt('transformed_data.csv', delimiter=',')
    data=my_data[:,range(13)]
    labels=my_data[:,-1]
    return data,labels

data,labels=read_data()
pdb.set_trace()
