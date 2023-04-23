from ..mcs0 import teq_main as teq_main_mcs0


def teq_main(
        index: int,
        beam_cross_section_area: float,
        beam_position_vertical: float,
        # beam_position_horizontal,  # depreciated from mcs0
        beam_position_horizontal_ratio: float,  # new from mcs0
        beam_rho: float,
        fire_time_duration: float,
        fire_time_step: float,
        fire_combustion_efficiency: float,
        fire_gamma_fi_q: float,
        fire_hrr_density: float,
        fire_load_density: float,
        fire_mode: int,
        fire_nft_limit: float,
        fire_spread_speed: float,
        fire_t_alpha: float,
        fire_tlim: float,
        protection_c: float,
        protection_k: float,
        protection_protected_perimeter: float,
        protection_rho: float,
        # room_breadth: float,  # depreciated from mcs0
        # room_depth: float,  # depreciated from mcs0
        room_height: float,
        room_wall_thermal_inertia: float,
        room_floor_area: float,  # new from mcs0
        room_breadth_depth_ratio: float,  # new from mcs0
        solver_temperature_goal: float,
        solver_max_iter: int,
        solver_thickness_lbound: float,
        solver_thickness_ubound: float,
        solver_tol: float,
        # window_width: float,  # depreciated from mcs0
        window_open_fraction: float,
        # window_height: float,  # depreciated from mcs0
        window_open_fraction_permanent: float,
        window_height_room_height_ratio: float,  # new from mcs0
        window_area_floor_ratio: float,  # new from mcs0
        phi_teq: float = 1.0,
        timber_charring_rate=None,
        timber_charred_depth=None,
        timber_hc: float = None,
        timber_density: float = None,
        timber_exposed_area: float = None,
        timber_depth: float = None,
        timber_solver_tol: float = None,
        timber_solver_ilim: float = None,
        occupancy_type: str = None,
        car_cluster_size: int = None,
) -> tuple:
    # -----------------------------------------
    # Calculate `room_breadth` and `room_depth`
    # -----------------------------------------
    # room_depth * room_breadth = room_floor_area
    # room_breadth / room_depth = room_breadth_depth_ratio

    # room_breadth = room_breadth_depth_ratio * room_depth
    # room_depth * room_breadth_depth_ratio * room_depth = room_floor_area
    room_depth = (room_floor_area / room_breadth_depth_ratio) ** 0.5
    room_breadth = room_breadth_depth_ratio * (room_floor_area / room_breadth_depth_ratio) ** 0.5
    assert 0 < room_breadth_depth_ratio <= 1.  # ensure within (0, 1]
    assert abs(
        room_depth * room_breadth - room_floor_area) < 1e-5  # ensure calculated room floor dimensions match the prescribed floor area

    # -----------------------------------------
    # Calculate window opening width and height
    # -----------------------------------------
    window_height = window_height_room_height_ratio * room_height
    window_width = room_floor_area * window_area_floor_ratio / window_height
    assert 0 < window_height_room_height_ratio <= 1.  # ensure within (0, 1]

    # --------------------------------
    # Calculate beam vertical location
    # --------------------------------
    beam_position_vertical = min(beam_position_vertical, room_height)
    beam_position_horizontal=room_depth * beam_position_horizontal_ratio

    return (*teq_main_mcs0(
        index=index,
        beam_cross_section_area=beam_cross_section_area,
        beam_position_vertical=beam_position_vertical,
        beam_position_horizontal=beam_position_horizontal,
        beam_rho=beam_rho,
        fire_time_duration=fire_time_duration,
        fire_time_step=fire_time_step,
        fire_combustion_efficiency=fire_combustion_efficiency,
        fire_gamma_fi_q=fire_gamma_fi_q,
        fire_hrr_density=fire_hrr_density,
        fire_load_density=fire_load_density,
        fire_mode=fire_mode,
        fire_nft_limit=fire_nft_limit,
        fire_spread_speed=fire_spread_speed,
        fire_t_alpha=fire_t_alpha,
        fire_tlim=fire_tlim,
        protection_c=protection_c,
        protection_k=protection_k,
        protection_protected_perimeter=protection_protected_perimeter,
        protection_rho=protection_rho,
        room_breadth=room_breadth,
        room_depth=room_depth,
        room_height=room_height,
        room_wall_thermal_inertia=room_wall_thermal_inertia,
        solver_temperature_goal=solver_temperature_goal,
        solver_max_iter=solver_max_iter,
        solver_thickness_lbound=solver_thickness_lbound,
        solver_thickness_ubound=solver_thickness_ubound,
        solver_tol=solver_tol,
        window_height=window_height,
        window_open_fraction=window_open_fraction,
        window_width=window_width,
        window_open_fraction_permanent=window_open_fraction_permanent,
        phi_teq=phi_teq,
        timber_charring_rate=timber_charring_rate,
        timber_charred_depth=timber_charred_depth,
        timber_hc=timber_hc,
        timber_density=timber_density,
        timber_exposed_area=timber_exposed_area,
        timber_depth=timber_depth,
        timber_solver_tol=timber_solver_tol,
        timber_solver_ilim=timber_solver_ilim,
        occupancy_type=occupancy_type,
        car_cluster_size=car_cluster_size,
    ),)
