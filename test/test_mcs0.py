# -*- coding: utf-8 -*-

from sfeprapy.mcs0.mcs0_calc import _test_teq_phi as test_teq_phi
from sfeprapy.mcs0.mcs0_calc import _test_standard_case as test_mcs0_test_standard_case
from sfeprapy.mcs0.mcs0_calc import _test_standard_oversized_case as test_mcs0_test_standard_oversized_case

test_teq_phi()
test_mcs0_test_standard_case()
test_mcs0_test_standard_oversized_case()