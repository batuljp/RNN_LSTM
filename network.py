import numpy as np
from node import node

#Class that creates the network
class network_lstm():

    def __init__(self, paramters_of_lstm, final_loss):
        #parameters
        self.paramters_of_lstm = paramters_of_lstm
        #list to hold all cell nodes
        self.n_list = []
        #training data
        self.xlist = []
        #final_loss
        self.final_loss=final_loss
      

    #Function  to update the node list and get the final loss
    def update_total_differences(self, tru_lost):
        #retrieve data index
        position_of_data = len(self.xlist) - 1
      
        if(self.final_loss=='mae'):
            final_loss = (np.abs(self.n_list[position_of_data].param_six[0]-tru_lost[position_of_data]))
        else:
            final_loss = (self.n_list[position_of_data].param_six[0] - tru_lost[position_of_data]) ** 2

        total_difference = np.zeros_like(self.n_list[position_of_data].param_six)
        total_difference[0] =2*(self.n_list[position_of_data].param_six[0] - tru_lost[position_of_data])
        
        total_difference_h = total_difference

        # update the next nodes
        total_difference_s = np.zeros(self.paramters_of_lstm.layers)

        # geet total_differences for the laters
        diff_one = self.n_list[position_of_data].param_four * total_difference_h + total_difference_s
        diff_two = self.n_list[position_of_data].param_five * total_difference_h
        diff_three = self.n_list[position_of_data].param_one * diff_one
        diff_four = self.n_list[position_of_data].param_two * diff_one
        diff_five = self.n_list[position_of_data].prev_one * diff_one


        # calculate derivatives for the input states
        diff_three_input = self.n_list[position_of_data].param_two*(1-self.n_list[position_of_data].param_two) * diff_three 
        diff_five_input = self.n_list[position_of_data].param_two*(1-self.n_list[position_of_data].param_three) * diff_five 
        diff_two_input = self.n_list[position_of_data].param_two*(1-self.n_list[position_of_data].param_four) * diff_two 
        diff_four_input = (1. - self.n_list[position_of_data].param_one ** 2) * diff_four



        # update the total_differences or derivatives
        self.n_list[position_of_data].param.weight_two_difference += np.outer(diff_three_input, self.n_list[position_of_data].cross_join)
        self.n_list[position_of_data].param.weight_three_difference += np.outer(diff_five_input, self.n_list[position_of_data].cross_join)
        self.n_list[position_of_data].param.weight_four_difference += np.outer(diff_two_input, self.n_list[position_of_data].cross_join)
        self.n_list[position_of_data].param.weight_one_difference += np.outer(diff_four_input, self.n_list[position_of_data].cross_join)
        self.n_list[position_of_data].param.bias_two_weight += diff_three_input
        self.n_list[position_of_data].param.bias_three_weight += diff_five_input       
        self.n_list[position_of_data].param.bias_four_weight += diff_two_input
        self.n_list[position_of_data].param.bias_one_weight += diff_four_input

        # get derivative total_differenceerences
        d_cross_difference = np.zeros_like(self.n_list[position_of_data].cross_join)+ np.dot(self.n_list[position_of_data].param.weight_two.T, diff_three_input) + np.dot(self.n_list[position_of_data].param.weight_three.T, diff_five_input) + np.dot(self.n_list[position_of_data].param.weight_four.T, diff_two_input) + np.dot(self.n_list[position_of_data].param.weight_one.T, diff_four_input)
        
        #save these total_differencees to the cell state
        self.n_list[position_of_data].total_difference_bottom_s = diff_one * self.n_list[position_of_data].param_three
        self.n_list[position_of_data].total_difference_botton_h = d_cross_difference[self.n_list[position_of_data].param.X:]

        position_of_data = position_of_data - 1

        #iterate all indexes while getting total_differences for all nodes, add total_differences to dff_h
        while position_of_data >= 0:

            #calculate the total_differenceerennce as final_loss
            if(self.final_loss=='mae'):
                final_loss = (np.abs(self.n_list[position_of_data].param_six[0]-tru_lost[position_of_data]))
            else:
                final_loss = (self.n_list[position_of_data].param_six[0] - tru_lost[position_of_data]) ** 2
            
            total_difference = np.zeros_like(self.n_list[position_of_data].param_six)
            total_difference[0] =2*(self.n_list[position_of_data].param_six[0] - tru_lost[position_of_data])
            total_difference_h = total_difference

            total_difference_h = self.n_list[position_of_data + 1].total_difference_botton_h + total_difference_h
            total_difference_s = self.n_list[position_of_data + 1].total_difference_bottom_s

            # geet total_differences for the laters
            diff_one = (self.n_list[position_of_data].param_four * total_difference_h) + total_difference_s
            diff_two = self.n_list[position_of_data].param_five * total_difference_h
            diff_three = self.n_list[position_of_data].param_one * diff_one
            diff_four = self.n_list[position_of_data].param_two * diff_one
            diff_five = self.n_list[position_of_data].prev_one * diff_one

            # calculate derivatives for the input states
            diff_three_input = self.n_list[position_of_data].param_two*(1-self.n_list[position_of_data].param_two) * diff_three 
            diff_five_input = self.n_list[position_of_data].param_two*(1-self.n_list[position_of_data].param_three) * diff_five 
            diff_two_input = self.n_list[position_of_data].param_two*(1-self.n_list[position_of_data].param_four) * diff_two 
            diff_four_input = (1. - self.n_list[position_of_data].param_one ** 2) * diff_four

            # update the total_differences or derivatives
            self.n_list[position_of_data].param.weight_two_difference += np.outer(diff_three_input, self.n_list[position_of_data].cross_join)
            self.n_list[position_of_data].param.weight_three_difference += np.outer(diff_five_input, self.n_list[position_of_data].cross_join)
            self.n_list[position_of_data].param.weight_four_difference += np.outer(diff_two_input, self.n_list[position_of_data].cross_join)
            self.n_list[position_of_data].param.weight_one_difference += np.outer(diff_four_input, self.n_list[position_of_data].cross_join)
            self.n_list[position_of_data].param.bias_two_weight += diff_three_input
            self.n_list[position_of_data].param.bias_three_weight += diff_five_input       
            self.n_list[position_of_data].param.bias_four_weight += diff_two_input
            self.n_list[position_of_data].param.bias_one_weight += diff_four_input

            # get derivative total_differenceerences
            d_cross_difference = np.zeros_like(self.n_list[position_of_data].cross_join) + np.dot(self.n_list[position_of_data].param.weight_two.T, diff_three_input) + np.dot(self.n_list[position_of_data].param.weight_three.T, diff_five_input)  + np.dot(self.n_list[position_of_data].param.weight_four.T, diff_two_input) + np.dot(self.n_list[position_of_data].param.weight_one.T, diff_four_input)
            
            #save these total_differencees to the cell state
            self.n_list[position_of_data].total_difference_bottom_s = diff_one * self.n_list[position_of_data].param_three
            self.n_list[position_of_data].total_difference_botton_h = d_cross_difference[self.n_list[position_of_data].param.X:]
            position_of_data = position_of_data-  1 
            
        return final_loss




