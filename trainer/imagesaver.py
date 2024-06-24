import importlib


def get_function(function_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        func = getattr(m, function_name, None)
        if func is not None:
            return func
    return RuntimeError(f'unsupported function name:{function_name}')


def get_image_saver(config):
    modules_list = [
        'trainer.test.testmdel'
    ]

    try:
        funcs = get_function(config['name'], modules_list)
    except Exception as e:
        funcs = None
    return funcs
