# -*- coding: utf-8 -*-
import copy
import re


def kwargs_from_text(raw_input_text_str):

    # with open(input_file_dir) as input_file:
    #     input_text = input_file.read()

    input_text = copy.copy(raw_input_text_str)

    # remove spaces
    input_text = input_text.replace(" ", "")

    # remove tabs
    input_text = input_text.replace("\t", "")

    # break raw input text to a list
    input_text = re.split('[;\r\n]', input_text)

    # delete comments (anything followed by #)
    input_text = [v.split("#")[0] for v in input_text]

    # delete empty entries
    input_text = [v for v in input_text if v]

    # for each entry in the list (input_text), break up i.e. ["variable_name=1+1"] to [["variable_name"], ["1+1"]]
    input_text = [v.split("=") for v in input_text]

    # analyse for each individual element input and create a list of library
    dict_inputs = {}
    for each_entry in input_text:
        dict_inputs[each_entry[0]] = eval(each_entry[1])
    return dict_inputs


if __name__ == "__main__":
    inputs_string = """# this is a comment
    my_variable_1=2.5;
    my_variable_2 = 3e8  # speed of light [m/2];
    my_dictionary_1 ={"key1":1,"key2":2,"key3":"hello_string"};
    my_string= "hello_world";
    # end"""

    inputs_dict=kwargs_from_text(inputs_string)
    print(inputs_dict)