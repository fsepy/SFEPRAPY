def fire(time, growth_factor, cap_hrr=0, cap_hrr_to_time=0):
    if isinstance(growth_factor, str):
        growth_factor = str.lower(growth_factor)
        growth_dict = {
            "slow": 0.0029,
            "medium": 0.0117,
            "fast": 0.0469,
            "ultra-fast": 0.1876,
        }
        try:
            growth_factor = growth_dict[growth_factor]
        except KeyError:
            err_msg = "{} should be one of the following if not a number: {}"
            err_msg = err_msg.format(
                "growth_factor", ", ".join(list(growth_dict.keys()))
            )
            raise ValueError(err_msg)

    heat_release_rate = growth_factor * time ** 2 * 1000

    # cap hrr
    if cap_hrr:
        heat_release_rate[heat_release_rate > cap_hrr] = cap_hrr

    if cap_hrr_to_time:
        heat_release_rate[time > cap_hrr_to_time] = -1
        heat_release_rate[heat_release_rate == -1] = max(heat_release_rate)

    return heat_release_rate
