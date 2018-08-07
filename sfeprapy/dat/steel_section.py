# -*- coding: utf-8 -*-
# AUTHOR: YAN FU
# DATE: 23/01/2018
# FILE NAME: steel_carbon.py
# DESCRIPTION: This python script file contains SteelSection object which transforms data stored within section_UB.csv
#              to Python readable format. Details see description under the object.

import os
import pandas as pd
import numpy as np


class SteelSection(object):
    """
    DESCRIPTION: SteelSection aims to transform the data stored in .csv files (i.e. section_UB.csv and section_UC.csv)
    into a pythonic format, i.e. an object with properties.
    """
    def __init__(self, section_type, section_desination):
        # Load data file, i.e. a .csv file, and make it in DataFrame format
        dict_type_to_directory = {
            'ub': 'sections_UB.csv',
            'uc': 'sections_UC.csv',}

        file_name_enquired_property = dict_type_to_directory[section_type]

        dir_this_folder = os.path.dirname(os.path.abspath(__file__))

        dir_file = "/".join([dir_this_folder, file_name_enquired_property])

        # Check if the subject file exists. An file
        if not os.path.isfile(dir_file):
            raise FileNotFoundError("File does not exist: ".format(dir_file))

        self.__data_all = pd.read_csv(filepath_or_buffer=dir_file, header=2, index_col=0, dtype={'id': np.str})
        self.__data_selected = self.__data_all[self.__data_all.index == section_desination]

        '''
        depth of section,width of section,web thickness,flange thickness,root radius,depth between fillets,ratios for local buckling (web),
        ratios for local buckling (flange),
        
        dimensions for detailing (end clearance),dimensions for detailing (notch),dimensions for detailing (notch),
        
        '''

    def __extract_col_data(self, parameter):
        result = self.__data_selected[parameter].values
        if len(result) == 1:
            result = result[0]
        return result

    def mass_per_metre(self):
        pass

    def depth(self):
        pass

    def width(self):
        pass

    def thickness_web(self):
        pass

    def thickness_flange(self):
        pass

    def root_radius(self):
        pass

    def depth_between_fillets(self):
        pass

    def ratios_local_buckling_web(self):
        pass

    def ratios_local_buckling_flange(self):
        pass

    '''
    surface area per metre,second moment of area (y-y),second moment of area (z-z),radius of gyration (y-y),radius of gyration (z-z),elastic modulus (y-y),'''
    def surface_area_per_metre(self):
        pass

    def second_moment_of_area_yy(self):
        pass

    def second_moment_of_area_zz(self):
        pass

    def radius_of_gyration_yy(self):
        pass

    def elastic_modulus_yy(self):
        pass

    '''
    elastic modulus (z-z),plastic modulus (y-y),plastic modulus (z-z),buckling parameter,torsional index,warping constant,torsional constant,area of section
    '''

    def elastic_modulus_zz(self):
        pass

    def plastic_modulus_yy(self):
        pass

    def plastic_modulus_zz(self):
        pass

    def buckling_parameter(self):
        pass

    def torsional_index(self):
        pass

    def torsional_constant(self):
        pass

    @property
    def SECTION_TYPE_UB(self): return 'ub'

    @property
    def SECTION_TYPE_UC(self): return 'uc'


if __name__ == '__main__':
    ss = SteelSection
    my_section = ss('ub', 'h')
