"""Example inputs for ``teq_main``.

Each case is a dict of ``teq_main`` parameters. Entries that are ``dict(dist=...)``
describe how the parameter would be sampled in a Monte Carlo run; ``teq_main`` itself
does not consume them -- callers resolve them to concrete values (see
``test/test_mcs0.py::test_standard_case`` for a worked example).
"""

# Common parameters shared by every example case (a representative office compartment).
_BASE = dict(
    fire_time_step=10,
    fire_time_duration=18000,
    fire_hrr_density=dict(dist="uniform_", lbound=0.25 - 0.001, ubound=0.25 + 0.001),
    fire_load_density=dict(dist="gumbel_r_", lbound=10, ubound=1500, mean=420, sd=126),
    fire_spread_speed=dict(dist="uniform_", lbound=0.0035, ubound=0.0190),
    fire_nft_limit=dict(dist="norm_", lbound=623.15, ubound=1473.15, mean=1323.15, sd=93),
    fire_combustion_efficiency=dict(dist="uniform_", lbound=0.8, ubound=1.0),
    beam_cross_section_area=0.017,
    beam_position_horizontal=dict(dist="uniform_", lbound=0.6 * 31.25, ubound=0.9 * 31.25),
    beam_position_vertical=3.1,
    beam_rho=7850,
    fire_mode=3,
    fire_tlim=0.333,
    protection_c=1700,
    protection_k=0.2,
    protection_protected_perimeter=2.14,
    protection_rho=800,
    room_breadth=16,
    room_depth=31.25,
    room_height=3.1,
    room_wall_thermal_inertia=720,
    solver_temperature_goal=823.15,
    solver_tol=1.0,
    window_height=2.8,
    # open window width ~20% of the 16 m wall -> mean open width 14.4 m
    window_width=dict(dist="lognorm_mod_", ubound=72.0, lbound=0.0, mean=14.4, sd=14.4),
    timber_solver_ilim=20,
    timber_solver_tol=1,
)


def _case(**overrides):
    """Build an example case from the common base plus per-case overrides."""
    case = dict(_BASE)
    case.update(overrides)
    return case


EXAMPLE_INPUT = {
    # Standard case: no timber contribution.
    'CASE_1': _case(timber_burning_rate=0, timber_fire_load_max=None),
    # Second independent sample of the standard case (was differentiated by a
    # model-uncertainty factor `phi_teq`, since removed).
    'CASE_2': _case(timber_burning_rate=0, timber_fire_load_max=None),
    # Timber case: timber contributes energy at 30.8 MJ/s (= 400 kg/m3 * 13.2 MJ/kg
    # * 500 m2 * 0.7 mm/min), the prior reference timber configuration.
    'CASE_3_timber': _case(
        timber_burning_rate=30.8,
        timber_fire_load_max=None,  # MJ (None = no cap)
    ),
}
