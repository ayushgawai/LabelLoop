def instruction(state, cfg, **kw):
    options = cfg["instructions"][state]
    return options[cfg["instruction_variant"] % len(options)].format(**kw)
