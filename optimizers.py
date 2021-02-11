from torch import optim

def get_optim_from_config(model_params, config):
    assert len(config['training']['optimizer']) == 1,'Only one optimizer can be given'
    print(list(config['training']['optimizer'].items()))
    optim_name, optim_kwargs = list(config['training']['optimizer'].items())[0]
    optim_fn = getattr(optim, optim_name)
    optimizer = optim_fn(model_params, **optim_kwargs)
    print(optimizer)
    return optimizer

def get_scheduler_from_config(optimizer, config):
    assert len(config['training']['scheduler']) == 1,'Only one scheduler can be given'
    sched_name, sched_kwargs = list(config['training']['scheduler'].items())[0]
    sched_fn = getattr(optim.lr_scheduler, sched_name)
    sched = sched_fn(optimizer, **sched_kwargs)
    print(sched)
    return sched