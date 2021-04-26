# -*- coding: utf-8 -*-
import pandas as pd

from sfeprapy.mcs.mcs_gen import dict_flatten


def __example_input_list() -> list:
    y = [
        dict(
            case_name="Standard Case 1",
            n_simulations=2500,
            fire_time_step=10,
            fire_time_duration=18000,
            fire_hrr_density=dict(dist="uniform_", lbound=0.25 - 0.001, ubound=0.25 + 0.001),
            fire_load_density=dict(dist="gumbel_r_", lbound=10, ubound=1500, mean=420, sd=126),
            fire_spread_speed=dict(dist="uniform_", lbound=0.0035, ubound=0.0190),
            fire_nft_limit=dict(dist="norm_", lbound=623.15, ubound=1473.15, mean=1323.15, sd=93),
            fire_combustion_efficiency=dict(dist="uniform_", lbound=0.8, ubound=1.0),
            window_open_fraction=dict(dist="lognorm_mod_", ubound=0.9999, lbound=0.0001, mean=0.2, sd=0.2),
            phi_teq=dict(dist="constant_", ubound=1, lbound=1, mean=0, sd=0),
            beam_cross_section_area=0.017,
            beam_position_horizontal=dict(dist="uniform_", lbound=0.6 * 31.25, ubound=0.9 * 31.25),
            beam_position_vertical=3.2,
            beam_rho=7850,
            fire_mode=3,
            fire_gamma_fi_q=1,
            fire_t_alpha=300,
            fire_tlim=0.333,
            protection_c=1700,
            protection_k=0.2,
            protection_protected_perimeter=2.14,
            protection_rho=800,
            room_breadth=16,
            room_depth=31.25,
            room_height=3.3,
            room_wall_thermal_inertia=720,
            solver_temperature_goal=823.15,
            solver_max_iter=20,
            solver_thickness_lbound=0.0001,
            solver_thickness_ubound=0.0400,
            solver_tol=1.0,
            window_height=2.8,
            window_width=72,
            window_open_fraction_permanent=0,
            timber_exposed_area=0,
            timber_charring_rate=0.7,  # mm/min
            timber_hc=13.2,  # MJ/kg
            timber_density=400,  # [kg/m3]
            timber_solver_ilim=20,
            timber_solver_tol=1,
            p1=3e-7,
            p2=0.1,
            p3=0.25,
            p4=0.09,
            general_room_floor_area=500,
        ),
        dict(
            case_name="Standard Case 2 (with teq_phi)",
            n_simulations=2500,
            fire_time_step=10,
            fire_time_duration=18000,
            fire_hrr_density=dict(dist="uniform_", lbound=0.25 - 0.001, ubound=0.25 + 0.001),
            fire_load_density=dict(dist="gumbel_r_", lbound=10, ubound=1500, mean=420, sd=126),
            fire_spread_speed=dict(dist="uniform_", lbound=0.0035, ubound=0.0190),
            fire_nft_limit=dict(dist="norm_", lbound=623.15, ubound=1473.15, mean=1323.15, sd=93),
            fire_combustion_efficiency=dict(dist="uniform_", lbound=0.8, ubound=1.0),
            window_open_fraction=dict(dist="lognorm_mod_", ubound=0.9999, lbound=0.0001, mean=0.2, sd=0.2),
            phi_teq=dict(dist="lognorm_", ubound=3, lbound=0.00001, mean=1, sd=0.25),
            beam_cross_section_area=0.017,
            beam_position_horizontal=dict(dist="uniform_", lbound=0.6 * 31.25, ubound=0.9 * 31.25),
            beam_position_vertical=3.2,
            beam_rho=7850,
            fire_mode=3,
            fire_gamma_fi_q=1,
            fire_t_alpha=300,
            fire_tlim=0.333,
            protection_c=1700,
            protection_k=0.2,
            protection_protected_perimeter=2.14,
            protection_rho=800,
            room_breadth=16,
            room_depth=31.25,
            room_height=3.3,
            room_wall_thermal_inertia=720,
            solver_temperature_goal=823.15,
            solver_max_iter=20,
            solver_thickness_lbound=0.0001,
            solver_thickness_ubound=0.0400,
            solver_tol=1.0,
            window_height=2.8,
            window_width=72,
            window_open_fraction_permanent=0,
            timber_exposed_area=0,
            timber_charring_rate=0.7,  # [mm/min]
            timber_hc=13.2,  # [MJ/kg]
            timber_density=400,  # [kg/m3]
            timber_solver_ilim=20,
            timber_solver_tol=1,
            p1=3e-7,
            p2=0.1,
            p3=0.25,
            p4=0.09,
            general_room_floor_area=500,
        ),
        dict(
            case_name="Standard Case 3 (with timber)",
            n_simulations=2500,
            fire_time_step=10,
            fire_time_duration=18000,
            fire_hrr_density=dict(dist="uniform_", lbound=0.25 - 0.001, ubound=0.25 + 0.001),
            fire_load_density=dict(dist="gumbel_r_", lbound=10, ubound=1500, mean=420, sd=126),
            fire_spread_speed=dict(dist="uniform_", lbound=0.0035, ubound=0.0190),
            fire_nft_limit=dict(dist="norm_", lbound=623.15, ubound=1473.15, mean=1323.15, sd=93),
            fire_combustion_efficiency=dict(dist="uniform_", lbound=0.8, ubound=1.0),
            window_open_fraction=dict(dist="lognorm_mod_", ubound=0.9999, lbound=0.0001, mean=0.2, sd=0.2),
            phi_teq=dict(dist="constant_", ubound=1, lbound=1, mean=0, sd=0),
            beam_cross_section_area=0.017,
            beam_position_horizontal=dict(dist="uniform_", lbound=0.6 * 31.25, ubound=0.9 * 31.25),
            beam_position_vertical=3.2,
            beam_rho=7850,
            fire_mode=3,
            fire_gamma_fi_q=1,
            fire_t_alpha=300,
            fire_tlim=0.333,
            protection_c=1700,
            protection_k=0.2,
            protection_protected_perimeter=2.14,
            protection_rho=800,
            room_breadth=16,
            room_depth=31.25,
            room_height=3.3,
            room_wall_thermal_inertia=720,
            solver_temperature_goal=823.15,
            solver_max_iter=20,
            solver_thickness_lbound=0.0001,
            solver_thickness_ubound=0.0400,
            solver_tol=1.0,
            window_height=2.8,
            window_width=72,
            window_open_fraction_permanent=0,
            timber_exposed_area=500.,
            timber_charring_rate=0.7,  # mm/min
            timber_hc=13.2,  # MJ/kg
            timber_density=400,  # [kg/m3]
            timber_solver_ilim=20,
            timber_solver_tol=1,
            p1=3e-7,
            p2=0.1,
            p3=0.25,
            p4=0.09,
            general_room_floor_area=500,
        ),
    ]
    return y


def __example_input_dict(x: list) -> dict:
    y = dict()
    for i in x:
        y[i['case_name']] = i
    return y


def __example_input_csv(x: list):
    y = [dict_flatten(v) for v in x]
    y = pd.DataFrame(y)
    y = y.transpose()
    y.index.name = "case_name"
    y = y.to_csv(index=True, line_terminator='\n')
    return y


def __example_input_df(x: list) -> pd.DataFrame:
    y = dict()
    for d in x:
        case_name = d.pop('case_name')
        d = dict_flatten(d)
        y[case_name] = d
    y = pd.DataFrame.from_dict(y)
    y.index.name = "case_name"
    return y


EXAMPLE_INPUT_DICT = __example_input_dict(x=__example_input_list())
EXAMPLE_INPUT_CSV = __example_input_csv(x=__example_input_list())
EXAMPLE_INPUT_DF = __example_input_df(x=__example_input_list())

if __name__ == "__main__":
    print(EXAMPLE_INPUT_DICT, "\n")
    print(EXAMPLE_INPUT_CSV, "\n")
