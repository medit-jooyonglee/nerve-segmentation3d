import os

import ml_collections
import yaml

from . import diskmanager
from .utils import get_logger

logger = get_logger(__name__)
def find_config(config_path_or_file, root):
    # logger =
    ext = os.path.splitext(config_path_or_file)[-1]
    if ext == '.yaml':
        return config_path_or_file
    else:
        root = os.path.dirname(root) if os.path.isfile(root) else root
        path = os.path.join(root, config_path_or_file)
        found = diskmanager.deep_serach_files(path, exts=['.yaml'])
        if not found:
            logger.error(f'not found yaml file:{path=}')
        # assert len(found) > 0,
        fname = lambda x: os.path.splitext(os.path.basename(x))[0]
        if len(found) > 1:
            # 최신순으로 정렬. 파일명 규칙 config_시간.yaml
            try:
                found = sorted(found, key=lambda v: -int(fname(v).split('_')[-1]))
            except Exception as e:
                found = [found[0]]
            # sorted(x, key=lambda v: int(v))
        return found[0] if found else ''





def load_config(filename='3DUnet_multiclass/train_config.yaml', root=__file__):
    """
    절대경로로 입력할 경후 해당 파일을 로드.
    상대경로로 파일을 입력할 경구 dataset/resources 경로에서 파일 로드
    Parameters
    ----------
    filename :

    Returns
    -------

    """
    # root =
    root = os.path.dirname(root) if os.path.isfile(root) else root

    # path = os.path.dirname(os.path.realpath(__file__))
    pathname = os.path.join(root, filename)
    pathname = os.path.realpath(pathname)
    pathname = filename if os.path.isabs(filename) else pathname

    assert os.path.exists(pathname), 'emtpy config yaml file:{}'.format(pathname)
    # trim path
    # logger = get_runtime_logger()

    logger.info('loading config file:{}'.format(pathname))
    # parser = argparse.ArgumentParser(description='UNet3D')
    # parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    # args = parser.parse_args()
    config = yaml.safe_load(open(pathname, 'r', encoding='utf-8'))

    return ml_collections.ConfigDict(config)

