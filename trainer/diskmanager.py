import json
import os
import shutil
from typing import List
import numpy as np

def deep_serach_files(basepath, exts=[]):
    """
    파일 딥 서치
    Parameters
    ----------
    basepath :
    exts :

    Returns
    -------

    """
    searched = []

    for root, dirs, files in os.walk(basepath):
        for ext in exts:
            file_pathes = [os.path.join(root, file) for file in files if isinstance(ext, str) and file.endswith(ext)]
            searched.extend(file_pathes)
    return searched

def deep_search_directory(basepath, exts=[], filter_func=None, strict=False, return_files=False):
    """
    디렉토리 딥 서치
    Parameters
    ----------
    basepath :
    exts :
    filter_func : 디렉토리에 해당 파일 목록에 대해 유효성 검사하는 callback 함수
    strict : bool
        디렉토리에 다른 확장자 파일이 있는지에 대한 옵션. true이면 해당 확장자가 섞여있을 경우 pass.
    return_files : bool
        default-false - 탐색한 파일목록도 반환 유무
    Returns
    -------
    list[str] the list of the directory
    list[list[str]] lost of files-list. optional
    """

    # logger = get_runtime_logger()
    searched = []
    files_list = []
    for root, dirs, files in os.walk(basepath):
        for ext in exts:
            filter_files = [file for file in files if isinstance(ext, str) and file.endswith(ext)]

            if strict and len(filter_files) == len(files):
                print(f'stride mode passed.- not same {ext} files#{len(filter_files)} = all files #{len(files)}')
                pass
            else:
                trim_path = os.path.realpath(root)
                if not trim_path in searched:
                    if filter_func is None:
                        searched.append(trim_path)
                        files_list.append(filter_files)
                    else:
                        if filter_func(filter_files):
                            searched.append(trim_path)
                            files_list.append(filter_files)
    if not return_files:
        return searched
    else:
        return (searched, files_list)


def get_size(path='.'):
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        return get_dir_size(path)

def get_dir_size(path='.'):
    """
    https://note.nkmk.me/en/python-os-path-getsize/
    Parameters
    ----------
    path :

    Returns
    -------

    """
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def move_files(source_files, source_base_path, target_bases):

    for src_path in source_files:
        relative = os.path.relpath(src_path, source_base_path)
        target_path = os.path.join(target_bases, relative)
        if not os.path.exists(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            print('empty move. target:\nfrom {}\n to {}'.format(src_path, target_path))
            shutil.move(src_path, target_path)
        elif get_size(src_path) != get_size(target_path):
            print('*****overwrite*****:\nfrom {}\n to {}'.format(src_path, target_path))
            shutil.move(src_path, target_path)
            # overwrite if exists
        else:
            print('directory all same size. passed move')
            pass


def copy_files(source_files, source_base_path, target_bases):

    for i, src_path in enumerate(source_files):
        relative = os.path.relpath(src_path, source_base_path)
        target_path = os.path.join(target_bases, relative)
        if not os.path.exists(target_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            print('empty copy. target:\nfrom {}\n to {}'.format(src_path, target_path))
            shutil.copy(src_path, target_path)
        elif get_size(src_path) != get_size(target_path):
            print('*****overwrite*****:\nfrom {}\n to {}'.format(src_path, target_path))
            shutil.copy(src_path, target_path)
            # overwrite if exists
        else:
            print('directory all same size. passed copy')
            pass
        print('progress....{}'.format(i/len(source_files)))


def move_directory(source_paths, source_base_path, target_bases):
    """
    source_base_path  베이스로 검색한 디렉토리 source_paths를
    target_bases 경로로 이동한다.
    이동할 각각의 target_path 는 source_paths 와 source_base_path 의 동일한 하위경로가 되도록 설정한다.

    Parameters
    ----------
    source_paths :
    source_base_path :
    target_bases :

    Returns
    -------

    """

    for src_path in source_paths:
        relative = os.path.relpath(src_path, source_base_path)
        target_path = os.path.join(target_bases, relative)
        if not os.path.exists(target_path):
            print('empty move. target:\nfrom {}\n to {}'.format(src_path, target_path))
            shutil.move(src_path, target_path)
        elif get_dir_size(src_path) != get_dir_size(target_path):
            print('*****overwrite*****:\nfrom {}\n to {}'.format(src_path, target_path))
            shutil.move(src_path, target_path)
            # overwrite if exists
        else:
            print('directory all same size. passed move')
            pass

        # shutil.move()

# def copy_file_d

def create_empty_directory(source_paths_or_files, source_base_path, target_bases):
    """
    source_base_path  베이스로 검색한 디렉토리 source_paths를
    target_bases 에 동일한 하위디렉토리 구조를 가지도록 빈 폴더를 생성한다.

    Parameters
    ----------
    source_paths :
    source_base_path :
    target_bases :

    Returns
    -------

    """

    for src_path_or_file in source_paths_or_files:
        src_path = os.path.dirname(src_path_or_file) if os.path.isfile(src_path_or_file) else src_path_or_file
        relative = os.path.relpath(src_path, source_base_path)
        target_path = os.path.join(target_bases, relative)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            print('create directory:\n{}'.format(target_path))
        # if not os.path.exists(target_path):
        #     print('empty move. target:\nfrom {}\n to {}'.format(src_path, target_path))
        #     shutil.move(src_path, target_path)
        # elif get_dir_size(src_path) != get_dir_size(target_path):
        #     print('*****overwrite*****:\nfrom {}\n to {}'.format(src_path, target_path))
        #     shutil.move(src_path, target_path)
        #     # overwrite if exists
        # else:
        #     print('directory all same size. passed move')
        #     pass


def split_dir_paths(pathname_or_filename):
    name = pathname_or_filename
    res = []
    while 1:
        nextname = os.path.dirname(name)
        # print(nextname)
        if name == nextname:
            if name:
                res.append(name)
            break
        else:
            res.append(os.path.basename(name))
        name = nextname

    return list(reversed(res))


def save_json(filename, founded):

    # https://jsikim1.tistory.com/222
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({'list': founded}, f, indent='\t', ensure_ascii=False)


def main_file_list_write():
    """
    데이터 목록 text 파일로 저장
    Returns
    -------

    """
    # path = '//192.168.98.132/f/DataSet/segmentation[reluapp]'
    path = '//192.168.98.132/d/DataSet/segmentation[reluapp]'
    filter_func = lambda x : len(x) > 10
    # 디렉토리 검색(ct같이 디렉토리로 관리되는 것들)
    founded = deep_search_directory(path, ['.ply'], filter_func)
    print(f'founded #{len(founded)}')
    filename = os.path.join(path, 'list.json')

    save_json(filename, founded)


def duplicate_check():
    path1 = 'D:/dataset/AutoPlanning(Nerve_Seg)/convert/source'
    path2 = 'D:/dataset/reluapp/source'

    found1 = deep_serach_files(path1, exts=['.npy'])
    found2 = deep_serach_files(path2, exts=['.npy'])

    fname1 = [os.path.basename(p) for p in found1]
    fname2 = [os.path.basename(p) for p in found2]
    equal = np.array(fname1)[:, None] == np.array(fname2)[None]
    equal_arg, = np.where(np.any(equal, axis=-1))
    print(equal_arg)
    print(equal_arg.size)
    print(np.array(fname1)[equal_arg])


def filter_path_keys(paths:List[str], keys:List[str]=[], reduce=any) -> List[str]:
    """

    Args:
        path (List[str]): 검사할 경로
        keys (List[str]): 특정 필터링 할 키 목록
        reduce (): any or all


    Returns:
    filter_paths: List[str] 필터링 된 경로

    """
    if keys:
        filters_paths = []
        for file in paths:
            if reduce([file.find(key) >=0 for key in keys]):
                filters_paths.append(file)
        return filters_paths
    else:
        return paths

def main():
    # FIXME: source path and target path
    # path = '//192.168.98.132/d/DataSet/CT_SCAN_RAW_DATA/marker_set(only2class)'
    # path = '//192.168.98.132/d/DataSet/autoplanning_design'
    # target = 'D:/dataset/autoplanning_design/marker_set(only2class)'
    # path = 'D:/dataset/autoplanning_gan_roi/split_and_crop'
    # path = '//192.168.98.132/d/DataSet/AutoPlanning(Nerve_Seg)'
    # path = 'D:/dataset/AutoPlanning(Nerve_Seg)/convert'
    path = 'D:/dataset/AutoPlanning(Nerve_Seg)/testset'
    # path = 'D:/dataset/reluapp'
    # path = 'C:/dataset/autoplanning_gan/test'
    # path - 'C:/dataset/ct_segmentation'
    # path = 'D:/datase/tTeethSegmentation[CBCT]'


    targets_list = [
        # 'D:/dataset/autoplanning_design',
        # 'D:/dataset/AutoPlanning(Nerve_Seg)',
        # '//192.168.98.132/f/Dataset/segmentation[reluapp]/npy'
        # '//192.168.98.211/c/dataset/autoplanning_gan/split_and_crop',
        # '//192.168.98.211/c/dataset/nerve_dataset/reluapp',
        '//192.168.98.211/c/dataset/nerve_dataset/valid',
        # '//192.168.98.132/c_drive_dataset/autoplanning_gan_roi/autoplanning_gan_roi/split_and_crop',

        # '//192.168.98.132/c_drive_dataset/autoplanning_gan_roi/test',
        # '//192.168.98.211/c/dataset/ct_segmentation',
        # '//192.168.98.211/c/dataset/autoplanning_gan/test',
        # '//192.168.98.132/c_drive_dataset/autoplanning_gan_roi/autoplanning_gan_roi/split_and_crop',
    ]

    # target = '//192.168.98.132/c_drive_dataset/autoplanning_gan_roi/test'
    # target = 'D:/dataset/autoplanning_design'

    filter_func = lambda x : len(x) > 10

    # 디렉토리 검색(ct같이 디렉토리로 관리되는 것들)
    # founded = deep_search_directory(path, ['.npy'])
    # 파일 검색(stl, mesh 데이터 등과 같이 파일단위로 관리되는 것들)
    founded = deep_serach_files(path, ['.npy'])
    print('founded', len(founded))

    # target 디렉토리로 데이터 이동
    # move_directory(founded, path, target)
    # target 디렉토리롤 파일 이동
    # move_files(founded, path, target)
    # target 디렉토리로 파일 복사(중복 체크)
    for target in targets_list:
        copy_files(founded, path, target)

    # target 디렉토리로 빈 폴더 생성
    # create_empty_directory(founded, path, target)


if __name__=='__main__':
    main()

    # found = deep_serach_files('//192.168.98.132/d/DataSet/TeethSegmentation[CBCT]/itk_snap_dataset/npy', exts=['.npy'])
    # print(f'found#{len(found)}')
    # main_file_list_write()
    # duplicate_check()
# def directory_move(source, dest, ext):

