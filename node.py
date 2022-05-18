import numpy as np

#Node Class for a single LSTM Node    
class node:
    def __init__(self, lstm_param, param_one, param_two, param_three, param_four, param_five, param_six, diff_bottom_h, diff_bottom_s):
        #paramaters
        self.param=lstm_param
        self.param_one=param_one
        self.param_two= param_two
        self.param_three=param_three
        self.param_four=param_four
        self.param_five=param_five
        self.param_six=param_six
        #joining all inputs
        self.cross_join = None

    #fxn to find data in bottom
    def data_bottom(self, x, prev_one=None, prev_two=None):
        # checking if this is or isnt first ndoe
        if(prev_one is None): 
            prev_one = np.zeros_like(self.param_five)
        if(prev_two is None): 
            prev_two = np.zeros_like(self.param_six)
        #save the information for later
        self.prev_one = prev_one
        self.prev_two = prev_two
        hat_cross_var = np.hstack((x,  prev_two))
        #get tanh value using function
        tan_h=(np.dot(self.param.weight_one, hat_cross_var) + self.param.bias_one_weight)
        #get dot product to pe used in sigmoid function
        sigmoid_1 = -(np.dot(self.param.weight_two, hat_cross_var) + self.param.bias_two_weight)
        sigmoid_2 = -(np.dot(self.param.weight_three, hat_cross_var) + self.param.bias_three_weight)
        sigmoid_3 = -(np.dot(self.param.weight_four, hat_cross_var) + self.param.bias_four_weight)
        # using sigmoid and tan join them
        self.param_one = np.tanh(tan_h)
        self.param_two = 1/(1+np.exp(sigmoid_1))
        self.param_three = 1/(1+np.exp(sigmoid_2))
        self.param_four = 1/(1+np.exp(sigmoid_3))
        self.param_five = (self.param_one*self.param_two)+(prev_one*self.param_three)
        self.param_six = (self.param_five*self.param_four)
        self.cross_join = hat_cross_var

