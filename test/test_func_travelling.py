# -*- coding: utf-8 -*-


from sfeprapy.func.fire_travelling import _test_fire as test_fire_travelling
from sfeprapy.func.fire_travelling import (
    _test_fire_backup as test_fire_travelling_backup,
)
from sfeprapy.func.fire_travelling import (
    _test_fire_multiple_beam_location as test_fire_travelling_multiple,
)

test_fire_travelling()
test_fire_travelling_backup()
test_fire_travelling_multiple()
