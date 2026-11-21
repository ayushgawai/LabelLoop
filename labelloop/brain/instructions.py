def instruction(state, cfg, variant=0, **kw):
    options = cfg["instructions"][state]
    return options[(cfg["instruction_variant"] + variant) % len(options)].format(**kw)
