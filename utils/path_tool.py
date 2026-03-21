# 为整个工程提供统一的绝对路径
import os


def get_prj_path():
    """
    获取工程根目录
    :return:
    """
    # 获取当前文件路径
    path = os.path.abspath(__file__)
    # 获取项目路径
    prj_path = os.path.dirname(os.path.dirname(path))
    return prj_path


def get_abs_path(file_path):
    """
    根据相对路径获取绝对路径
    :param file_path: 相对路径
    :return:
    """
    prj_path = get_prj_path()
    abs_path = os.path.join(prj_path, file_path)
    return abs_path