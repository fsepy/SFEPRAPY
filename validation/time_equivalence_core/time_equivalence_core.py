

def time_equivalence_core():

    import numpy as np
    from sfeprapy.time_equivalence_mc import calc_time_equivalence
    from sfeprapy.func.temperature_fires import standard_fire_iso834 as _fire_standard
    from sfeprapy.dat.steel_carbon import Thermal

    steel_prop = Thermal()
    beam_c = steel_prop.c()
    fire = _fire_standard(np.arange(0, 3*60*60, 1), 273.15+20)

    input_parameters = {
        "time_step": 1,
        "time_limiting": 0.333,
        "window_height": 2.8,
        "window_width": 72,
        "window_open_fraction": 0.8,
        "room_breadth": 15.8,
        "room_depth": 31.6,
        "room_height": 2.8,
        "room_wall_thermal_inertia": 720,
        "fire_load_density": 420,
        "fire_hrr_density": 0.25,
        "fire_spread_speed": 0.0114,
        "fire_duration": 18000,
        "beam_position": 0.75,
        "beam_rho": 7850,
        "beam_c": beam_c,
        "beam_cross_section_area": 0.017,
        "beam_temperature_goal": 620+273.15,
        "protection_k": 0.2,
        "protection_rho": 800,
        "protection_c": 1700,
        "protection_thickness": 0.0125,
        "protection_protected_perimeter": 2.14,
        "iso834_time": fire[0],
        "iso834_temperature": fire[1],
        "seek_ubound_iter": 20,
        "seek_tol_y": 1.,
        "nft_ubound": 1200,
        "return_mode": 2
    }

    results = calc_time_equivalence(**input_parameters)

    str_fmt = 'Match {:44}: {}'

    # check output set 1, input parameters
    benchmark_set1 = {
        'room_wall_thermal_inertia': 720,
        'fire_load_density': 420,
        'fire_hrr_density': 0.25,
        'fire_spread_speed': 0.0114,
        'fire_duration': 18000,
        'beam_position': 0.75,
        'beam_rho': 7850,
        'beam_cross_section_area': 0.017,
        'beam_temperature_goal': 893.15,
        'protection_k': 0.2,
        'protection_rho': 800,
        'protection_c': 1700,
        'protection_thickness': 0.0125,
        'protection_protected_perimeter': 2.14,
        'nft_ubound': 1200,
        'seek_ubound_iter': 20,
        'seek_tol_y': 1.0,
    }

    for key, val in benchmark_set1.items():
        print(str_fmt.format(key, val == input_parameters[key]))

    # check set 2, derived variables
    benchmark_set2 = {
        'index': -1,
        'fire_resistance_equivalence': 2413.08066401,
        'seek_status': True,
        'fire_type': 1,
        'sought_temperature_steel_ubound': 893.2451301167116,
        'sought_protection_thickness': 0.007368115234375001,
        'seek_count_iter': 11,
        'temperature_steel_ubound': int(771.7720741569833)
    }

    str_fmt = 'Check {:44}: {}'

    print(str_fmt.format('fire_type', benchmark_set2['fire_type'] == results['fire_type']))
    print(str_fmt.format('seek_status', benchmark_set2['seek_status'] == results['seek_status']))
    print(str_fmt.format('sought_temperature_steel_ubound', abs(input_parameters['beam_temperature_goal'] - results['sought_temperature_steel_ubound']) <= input_parameters['seek_tol_y']))

    return results


if __name__ == '__main__':
    time_equivalence_core()
