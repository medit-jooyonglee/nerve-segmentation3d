import ml_collections


from .utils import get_class

def get_model(model_config):
    model_class = get_class(model_config['name'], modules=[
        'trainer.test.testmodel',
        'interfaces.pidnetmodel',
        'vit_pytorch.vit',
        'relationattention',
        'models.model',
        'models.vit_seg_interpolator_modeling',
        # 'teethnet.models.unet3d.model',
        # 'planning_gan.models.transunet.transunet3d.vit_seg_modeling',
        # 'planning_gan.models.transunet.transunet3d.vit_seg_interpolator_modeling',
        # 'teethnet.models.pytorchyolo3d.model',
        # 'teethnet.models.relationobject.model',
        # 'teethnet.models.unetrpp.network_architecture.lung.unetr_pp_lung',
        # 'teethnet.models.voxelmorph.networks',
    ])
    try:
        return model_class(**model_config)
    except TypeError as e:
        arg = ml_collections.ConfigDict(model_config)
        return model_class(arg)
    except Exception as e:
        raise ValueError(e.args)
