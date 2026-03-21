import hashlib
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

from utils.logger_handler import logger


def get_file_md5_hex(filepath):
    if not os.path.exists(filepath):
        logger.error(f'[md5计算]{ filepath }文件不存在')

        return

    if not os.path.isfile(filepath):
        logger.error(f'[md5计算]{ filepath }不是文件')

        return

    md5_obj = hashlib.md5()

    # 以4096字节分片读取文件，计算md5
    chunk_size = 4096  # 4kb
    try:
        with open(filepath, "rb") as f:  # 必须二进制读取。rb指的是读取模式设定为二进制读取。
            while chunk := f.read(chunk_size):  # :=表示赋值，如果chunk为真，则执行
                md5_obj.update(chunk)
            """
               :=类似于以下
               chunk = f.read(chunk_size)
               while chunk:
                   md5_0bj.update(chunk)
                   chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            logger.info(f'[md5计算]{ filepath }文件md5计算成功，结果为{ md5_hex }')

            return md5_hex
    except Exception as e:
        logger.error(f'[md5计算]{ filepath }文件读取失败，{str(e)}')

        return None

# 找到指定目录下的指定后缀名的文件的路径，返回tuple[str]
# 其中后缀名以元组格式tuple[str]传入
def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    files = []

    if not os.path.isdir(path):
        logger.error(f'[文件列表]{ path }不是目录')

        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):  # endwith()方法用于判断字符串是否以指定后缀结尾，如果是则返回True，否则返回False
            files.append(os.path.join(path, f))

    return tuple(files)  # 返回tuple格式，避免列表被修改


def pdf_loader(filepath: str, pwd: str = None) -> list[Document]:
    return PyPDFLoader(filepath, pwd).load()


def txt_loader(filepath) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()


def docx_loader(filepath) -> list[Document]:
    """
    根据文件后缀名，自动选择加载器
    """
    return Docx2txtLoader(filepath).load()