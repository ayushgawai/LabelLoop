def instruction(state, cfg, bin=None):
    options = cfg["instructions"][state]
    return options[cfg["instruction_variant"] % len(options)].format(bin=bin)
