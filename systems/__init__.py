systems = {}


def register(name):
    '''一个很简单的装饰器, 用于注册网络, systems[name]就会指向对应的网络'''
    def decorator(cls):
        systems[name] = cls
        return cls
    return decorator


def make(name, config, load_from_checkpoint=None):
    '''构造指定的网络, Nerf或者Neus'''
    if load_from_checkpoint is None:
        system = systems[name](config)
    else:
        system = systems[name].load_from_checkpoint(load_from_checkpoint, strict=False, config=config)
    return system


from . import nerf, neus
