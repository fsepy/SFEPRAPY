# -*- coding: utf-8 -*-

from sfeprapy.mcs0.mcs0_test import test_arg as test_mcs0_test_arg
from sfeprapy.mcs0.mcs0_test import test_standard_case as test_mcs0_test_standard_case
from sfeprapy.func.mcs_gen import test_dict_flatten as test_mcs_gen_dict_flatten
from sfeprapy.func.mcs_gen import test_random_variable_generator as test_mcs_gen_random_variable_generator
from sfeprapy.func.fire_parametric_ec import test_fire as test_fire_parametric_ec
from sfeprapy.func.fire_travelling import test_fire as test_fire_travelling
from sfeprapy.func.fire_travelling import test_fire_backup as test_fire_travelling_backup
from sfeprapy.func.fire_travelling import test_fire_multiple_beam_location as test_fire_travelling_multiple

test_mcs0_test_arg()
test_mcs0_test_standard_case()
test_mcs_gen_dict_flatten()
test_mcs_gen_random_variable_generator()
test_fire_parametric_ec()
test_fire_travelling()
test_fire_travelling_backup()
test_fire_travelling_multiple()
