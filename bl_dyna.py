import numpy as np
import pandas as pd
from scipy.spatial import distance
import pathlib, copy
from types import MappingProxyType
from collections import defaultdict
from itertools import groupby
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

EntityCls_CardFields = {
    **{
        "*CONTROL_TERMINATION": [[10] * 6],
        "*CONTROL_TIMESTEP": [[10] * 8],
        "*DATABASE_ELOUT": [[10] * 8],
        "*DATABASE_GLSTAT": [[10] * 4],
        "*DATABASE_MATSUM": [[10] * 4],
        "*DATABASE_NODOUT": [[10] * 6],
        "*DATABASE_RCFORC": [[10] * 4],
        "*DATABASE_BINARY_D3PLOT": [[10] * 5],
        "*MAT_": [[10]],
        "*SECTION_": [[10]],
        "*SECTION_BEAM": [[10] * 8, [10] * 6],
        "*SECTION_SOLID_TITLE": [[80], [10]],
        "*SECTION_SHELL_TITLE": [[80], [10]],
        "*SECTION_DISCRETE_TITLE": [[80], [10]],
        "*SECTION_BEAM_TITLE": [[80], [10] * 8, [10] * 6],
        "*INCLUDE_": [[80]],
        "*INCLUDE_TRANSFORM": [[80], [10] * 7, [10] * 4, [10] * 5, [10] * 1],
        "*LOAD_NODE": [[10] * 8],
        "*PARAMETER": [[10] * 8],
        "*PARAMETER_EXPRESSION": [[10, 70]],
    },
    **{
        k: [[80], [10] * 8]
        for k in [
            "*MAT_LOW_DENSITY_FOAM_TITLE",
            "*MAT_PIECEWISE_LINEAR_PLASTICITY_TITLE",
            "*MAT_RIGID_TITLE",
            "*MAT_ELASTIC_TITLE",
            "*MAT_MODIFIED_PIECEWISE_LINEAR_PLASTICITY_TITLE",
            "*MAT_BLATZ-KO_RUBBER_TITLE",
            "*MAT_SPRING_ELASTIC_TITLE",
            "*MAT_DAMPER_VISCOUS_TITLE",
            "*MAT_NULL_TITLE",
            "*MAT_MODIFIED_HONEYCOMB_TITLE",
            "*MAT_SPRING_INELASTIC_TITLE",
        ]
    },
    **{
        "*NODE": [[8, 16, 16, 16, 8, 8]],
        "*ELEMENT_SOLID": [[8] * 10, [8] * 10],
        "*ELEMENT_SHELL": [[8] * 10],
        "*ELEMENT_SHELL_THICKNESS": [[8] * 10],
        "*ELEMENT_BEAM": [[8] * 10],
        "*ELEMENT_BEAM_OFFSET": [[8] * 10, [8] * 3],
        "*ELEMENT_BEAM_ORIENTATION": [[8] * 10, [8] * 6],
        "*PART": [[80], [10] * 8],
        "*DEFINE_CURVE": [[10] * 8, [20] * 2],
    },
    **{
        k: [[10] * 6, [10] * 8]
        for k in ["*SET_NODE", "*SET_PART", "*SET_SHELL"]
        + ["*SET_NODE_LIST", "*SET_PART_LIST", "*SET_SHELL_LIST"]
    },
}
EntityCls_PagmFields = {
    **{
        "*CONTROL_TERMINATION": {
            "ENDTIM": {"index": [0, 0], "format": ">+.{}e", "info": "控制终止时间"}
        },
        "*CONTROL_TIMESTEP": {
            "TSSFAC": {"index": [0, 1], "format": ">+.{}e", "info": "时间步长缩放系数"},
            "DT2MS": {"index": [0, 4], "format": ">+.{}e", "info": "人为控制时间步长"},
        },
        "*DATABASE_ELOUT": {
            "DT": {"index": [0, 0], "format": ">+.{}e", "info": "输出时间间隔"},
            "BINARY": {"index": [0, 1], "format": ">+{}d", "info": "使用二进制格式"},
        },
        "*DATABASE_GLSTAT": {
            "DT": {"index": [0, 0], "format": ">+.{}e", "info": "输出时间间隔"},
            "BINARY": {"index": [0, 1], "format": ">+{}d", "info": "使用二进制格式"},
        },
        "*DATABASE_MATSUM": {
            "DT": {"index": [0, 0], "format": ">+.{}e", "info": "输出时间间隔"},
            "BINARY": {"index": [0, 1], "format": ">+{}d", "info": "使用二进制格式"},
        },
        "*DATABASE_NODOUT": {
            "DT": {"index": [0, 0], "format": ">+.{}e", "info": "输出时间间隔"},
            "BINARY": {"index": [0, 1], "format": ">+{}d", "info": "使用二进制格式"},
        },
        "*DATABASE_RCFORC": {
            "DT": {"index": [0, 0], "format": ">+.{}e", "info": "输出时间间隔"},
            "BINARY": {"index": [0, 1], "format": ">+{}d", "info": "使用二进制格式"},
        },
        "*DATABASE_BINARY_D3PLOT": {
            "DT": {"index": [0, 0], "format": ">+.{}e", "info": "输出时间间隔"}
        },
        "*INCLUDE_": {"FILENAME": {"index": [0, 0], "format": "", "info": "文件名"}},
        "*INCLUDE_TRANSFORM": {
            "FILENAME": {"index": [0, 0], "format": "", "info": "文件名"},
            "IDNOFF": {"index": [1, 0], "format": "", "info": "节点 ID 偏移"},
            "IDEOFF": {"index": [1, 1], "format": "", "info": "元素 ID 偏移"},
            "IDPOFF": {"index": [1, 2], "format": "", "info": "部件 ID 偏移"},
            "IDMOFF": {"index": [1, 3], "format": "", "info": "材料 ID 偏移"},
            "IDSOFF": {"index": [1, 4], "format": "", "info": "截面 ID 偏移"},
            "IDFOFF": {"index": [1, 5], "format": "", "info": "集合 ID 偏移"},
            "IDDOFF": {"index": [1, 6], "format": "", "info": "曲线 ID 偏移"},
            "IDROFF": {"index": [2, 0], "format": "", "info": "其他 ID 偏移"},
            "PREFIX": {"index": [2, 2], "format": "", "info": "前缀"},
            "SUFFIX": {"index": [2, 3], "format": "", "info": "后缀"},
            "FCTMAS": {"index": [3, 0], "format": "", "info": "质量转换因子"},
            "FCTTIM": {"index": [3, 1], "format": "", "info": "时间转换因子"},
            "FCTLEN": {"index": [3, 2], "format": "", "info": "长度转换因子"},
            "FCTTEM": {"index": [3, 3], "format": "", "info": "温度转换因子"},
            "INCOUT1": {"index": [3, 4], "format": "", "info": "输出文件开关"},
            "TRANID": {"index": [4, 0], "format": "", "info": "转换定义关键字 ID"},
        },
        "*MAT_": {"MID": {"index": [0, 0], "format": ">+{}d", "info": "id"}},
        "*SECTION_": {"SECID": {"index": [0, 0], "format": ">+{}d", "info": "id"}},
        "*SECTION_BEAM": {
            "SECID": {"index": [0, 0], "format": ">+{}d", "info": "id"},
            "ELFORM": {"index": [0, 1], "format": ">+{}d", "info": "单元形状"},
            "SHRF": {"index": [0, 2], "format": ">+.{}e", "info": "截面高宽比"},
            "QR_IRID": {"index": [0, 3], "format": ">+{}d", "info": "弯矩法/惯性比"},
            "CST": {"index": [0, 4], "format": "", "info": "截面类型"},
            "SCCOOR": {"index": [0, 5], "format": "", "info": "截面坐标"},
            "NSM": {"index": [0, 6], "format": "", "info": "非结构质量"},
            "NAUPD": {"index": [0, 7], "format": "", "info": "非结构质量更新"},
            "TS1": {"index": [1, 0], "format": "", "info": "剪切刚度1"},
            "TS2": {"index": [1, 1], "format": "", "info": "剪切刚度2"},
            "TT1": {"index": [1, 2], "format": "", "info": "扭转刚度1"},
            "TT2": {"index": [1, 3], "format": "", "info": "扭转刚度2"},
            "NSLOC": {"index": [1, 4], "format": "", "info": "剪切中性轴位置"},
            "NTLOC": {"index": [1, 5], "format": "", "info": "扭转中性轴位置"},
        },
        "*SECTION_SOLID_TITLE": {"SECID": {"index": [1, 0], "format": ">+{}d", "info": "id"}},
        "*SECTION_SHELL_TITLE": {"SECID": {"index": [1, 0], "format": ">+{}d", "info": "id"}},
        "*SECTION_DISCRETE_TITLE": {"SECID": {"index": [1, 0], "format": ">+{}d", "info": "id"}},
        "*SECTION_BEAM_TITLE": {
            "SECID": {"index": [1 + 0, 0], "format": ">+{}d", "info": "id"},
            "ELFORM": {"index": [1 + 0, 1], "format": ">+{}d", "info": "单元形状"},
            "SHRF": {"index": [1 + 0, 2], "format": ">+.{}e", "info": "截面高宽比"},
            "QR_IRID": {"index": [1 + 0, 3], "format": ">+{}d", "info": "弯矩法/惯性比"},
            "CST": {"index": [1 + 0, 4], "format": "", "info": "截面类型"},
            "SCCOOR": {"index": [1 + 0, 5], "format": "", "info": "截面坐标"},
            "NSM": {"index": [1 + 0, 6], "format": "", "info": "非结构质量"},
            "NAUPD": {"index": [1 + 0, 7], "format": "", "info": "非结构质量更新"},
            "TS1": {"index": [1 + 1, 0], "format": "", "info": "剪切刚度1"},
            "TS2": {"index": [1 + 1, 1], "format": "", "info": "剪切刚度2"},
            "TT1": {"index": [1 + 1, 2], "format": "", "info": "扭转刚度1"},
            "TT2": {"index": [1 + 1, 3], "format": "", "info": "扭转刚度2"},
            "NSLOC": {"index": [1 + 1, 4], "format": "", "info": "剪切中性轴位置"},
            "NTLOC": {"index": [1 + 1, 5], "format": "", "info": "扭转中性轴位置"},
        },
        "*LOAD_NODE": {
            "ID": {"index": [0, 0], "format": "", "info": "节点ID或节点集ID"},
            "DOF": {
                "index": [0, 1],
                "format": "",
                "info": "适用的自由度 1:x 2:y 3:z 4:跟随 5:x矩 6:y矩 7:z矩 8:跟随矩",
            },
            "LCID": {"index": [0, 2], "format": "", "info": "曲线id"},
            "SF": {"index": [0, 3], "format": "", "info": "缩放比例"},
            "CID": {"index": [0, 4], "format": "", "info": "坐标系 可选"},
            "M1": {"index": [0, 5], "format": "", "info": "DOF=4/8可用"},
            "M2": {"index": [0, 6], "format": "", "info": "DOF=4/8可用"},
            "M3": {"index": [0, 7], "format": "", "info": "DOF=4/8可用"},
        },
        "*PARAMETER": {
            "K": {"index": [":", "::2"], "format": "", "info": "变量名"},
            "V": {"index": [":", "1::2"], "format": "", "info": "变量值"},
        },
        "*PARAMETER_EXPRESSION": {
            "K": {"index": [":", 0], "format": "", "info": "变量名"},
            "V": {"index": [":", 1], "format": "", "info": "变量值"},
        },
    },
    **{
        k: {
            "NAME": {"index": [0, 0], "format": "", "info": "title"},
            "MID": {"index": [1, 0], "format": "", "info": "id"},
        }
        for k in [
            "*MAT_LOW_DENSITY_FOAM_TITLE",
            "*MAT_PIECEWISE_LINEAR_PLASTICITY_TITLE",
            "*MAT_RIGID_TITLE",
            "*MAT_ELASTIC_TITLE",
            "*MAT_MODIFIED_PIECEWISE_LINEAR_PLASTICITY_TITLE",
            "*MAT_BLATZ-KO_RUBBER_TITLE",
            "*MAT_SPRING_ELASTIC_TITLE",
            "*MAT_DAMPER_VISCOUS_TITLE",
            "*MAT_NULL_TITLE",
            "*MAT_MODIFIED_HONEYCOMB_TITLE",
            "*MAT_SPRING_INELASTIC_TITLE",
        ]
    },
    **{
        "*NODE": {
            "ID": {"index": [":", 0], "format": "", "info": "id"},
            "X": {"index": [":", 1], "format": "", "info": "x坐标"},
            "Y": {"index": [":", 2], "format": "", "info": "y坐标"},
            "Z": {"index": [":", 3], "format": "", "info": "z坐标"},
            "TC": {"index": [":", 4], "format": "", "info": "温度"},
            "RC": {"index": [":", 5], "format": "", "info": "密度"},
        },
        "*ELEMENT_SOLID": {
            "ID": {"index": ["0::2", 0], "format": "", "info": "ID"},
            "PID": {"index": ["0::2", 1], "format": "", "info": "PART ID"},
            "N1": {"index": ["1::2", 0], "format": "", "info": "节点1"},
            "N2": {"index": ["1::2", 1], "format": "", "info": "节点2"},
            "N3": {"index": ["1::2", 2], "format": "", "info": "节点3"},
            "N4": {"index": ["1::2", 3], "format": "", "info": "节点4"},
            "N5": {"index": ["1::2", 4], "format": "", "info": "节点5"},
            "N6": {"index": ["1::2", 5], "format": "", "info": "节点6"},
            "N7": {"index": ["1::2", 6], "format": "", "info": "节点7"},
            "N8": {"index": ["1::2", 7], "format": "", "info": "节点8"},
        },
        "*ELEMENT_SHELL": {
            "ID": {"index": [":", 0], "format": "", "info": "ID"},
            "PID": {"index": [":", 1], "format": "", "info": "PART ID"},
            "N1": {"index": [":", 2], "format": "", "info": "节点1"},
            "N2": {"index": [":", 3], "format": "", "info": "节点2"},
            "N3": {"index": [":", 4], "format": "", "info": "节点3"},
            "N4": {"index": [":", 5], "format": "", "info": "节点4"},
            "N5": {"index": [":", 6], "format": "", "info": "节点5"},
            "N6": {"index": [":", 7], "format": "", "info": "节点6"},
            "N7": {"index": [":", 8], "format": "", "info": "节点7"},
            "N8": {"index": [":", 9], "format": "", "info": "节点8"},
        },
        "*ELEMENT_SHELL_THICKNESS": {
            "ID": {"index": ["::2", 0], "format": "", "info": "ID"},
            "PID": {"index": ["::2", 1], "format": "", "info": "PART ID"},
            "N1": {"index": ["::2", 2], "format": "", "info": "节点1"},
            "N2": {"index": ["::2", 3], "format": "", "info": "节点2"},
            "N3": {"index": ["::2", 4], "format": "", "info": "节点3"},
            "N4": {"index": ["::2", 5], "format": "", "info": "节点4"},
            "N5": {"index": ["::2", 6], "format": "", "info": "节点5"},
            "N6": {"index": ["::2", 7], "format": "", "info": "节点6"},
            "N7": {"index": ["::2", 8], "format": "", "info": "节点7"},
            "N8": {"index": ["::2", 9], "format": "", "info": "节点8"},
        },
        "*ELEMENT_BEAM": {
            "ID": {"index": [":", 0], "format": "", "info": "ID"},
            "PID": {"index": [":", 1], "format": "", "info": "PART ID"},
            "N1": {"index": [":", 2], "format": "", "info": "节点1"},
            "N2": {"index": [":", 3], "format": "", "info": "节点2"},
            "N3": {"index": [":", 4], "format": "", "info": "节点3"},
            "RT1": {"index": [":", 5], "format": "", "info": ""},
            "RR1": {"index": [":", 6], "format": "", "info": ""},
            "RT2": {"index": [":", 7], "format": "", "info": ""},
            "RR2": {"index": [":", 8], "format": "", "info": ""},
            "LOCAL": {"index": [":", 9], "format": "", "info": "局部坐标系"},
        },
        "*ELEMENT_BEAM_OFFSET": {
            "ID": {"index": ["0::2", 0], "format": "", "info": "ID"},
            "PID": {"index": ["0::2", 1], "format": "", "info": "PART ID"},
            "N1": {"index": ["0::2", 2], "format": "", "info": "节点1"},
            "N2": {"index": ["0::2", 3], "format": "", "info": "节点2"},
            "N3": {"index": ["0::2", 4], "format": "", "info": "节点3"},
            "RT1": {"index": ["0::2", 5], "format": "", "info": ""},
            "RR1": {"index": ["0::2", 6], "format": "", "info": ""},
            "RT2": {"index": ["0::2", 7], "format": "", "info": ""},
            "RR2": {"index": ["0::2", 8], "format": "", "info": ""},
            "LOCAL": {"index": ["0::2", 9], "format": "", "info": "局部坐标系"},
        },
        "*ELEMENT_BEAM_ORIENTATION": {
            "ID": {"index": ["0::2", 0], "format": "", "info": "ID"},
            "PID": {"index": ["0::2", 1], "format": "", "info": "PART ID"},
            "N1": {"index": ["0::2", 2], "format": "", "info": "节点1"},
            "N2": {"index": ["0::2", 3], "format": "", "info": "节点2"},
            "N3": {"index": ["0::2", 4], "format": "", "info": "节点3"},
            "RT1": {"index": ["0::2", 5], "format": "", "info": ""},
            "RR1": {"index": ["0::2", 6], "format": "", "info": ""},
            "RT2": {"index": ["0::2", 7], "format": "", "info": ""},
            "RR2": {"index": ["0::2", 8], "format": "", "info": ""},
            "LOCAL": {"index": ["0::2", 9], "format": "", "info": "局部坐标系"},
        },
        "*PART": {
            "NAME": {"index": ["0::2", 0], "format": "", "info": ""},
            "PID": {"index": ["1::2", 0], "format": "", "info": "PART ID"},
            "SECID": {"index": ["1::2", 1], "format": "", "info": "截面ID"},
            "MID": {"index": ["1::2", 2], "format": "", "info": "材料ID"},
            "EOSID": {"index": ["1::2", 3], "format": "", "info": "EOSID"},
            "HGID": {"index": ["1::2", 4], "format": "", "info": ""},
            "GRAV": {"index": ["1::2", 5], "format": "", "info": ""},
            "ADPOPT": {"index": ["1::2", 6], "format": "", "info": ""},
            "TMID": {"index": ["1::2", 7], "format": "", "info": ""},
        },
        "*DEFINE_CURVE": {
            "LCID": {"index": [0, 0], "format": "", "info": "曲线ID"},
            "SIDR": {"index": [0, 1], "format": "", "info": "曲线类型"},
            "SFA": {"index": [0, 2], "format": "", "info": "横轴缩放因子"},
            "SFO": {"index": [0, 3], "format": "", "info": "纵轴缩放因子"},
            "OFFA": {"index": [0, 4], "format": "", "info": "横轴偏移"},
            "OFFO": {"index": [0, 5], "format": "", "info": "纵轴偏移"},
            "DATTYP": {"index": [0, 6], "format": "", "info": "数据类型 留空默认为0"},
            "LCINT": {"index": [0, 7], "format": "", "info": "线性插值数量"},
            "X": {"index": ["1:", 0], "format": "", "info": "横轴数据"},
            "Y": {"index": ["1:", 1], "format": "", "info": "纵轴数据"},
        },
    },
    **{
        k: {
            "SID": {"index": [0, 0], "format": "", "info": "id"},
            "DA1": {"index": [0, 1], "format": "", "info": "数据1"},
            "DA2": {"index": [0, 2], "format": "", "info": "数据2"},
            "DA3": {"index": [0, 3], "format": "", "info": "数据3"},
            "DA4": {"index": [0, 4], "format": "", "info": "数据4"},
            "SLOVER": {"index": [0, 5], "format": "", "info": "求解器 默认MECH"},
            "NIDS": {"index": ["1:", ":"], "format": "", "info": "节点集"},
        }
        for k in ["*SET_NODE", "*SET_PART", "*SET_SHELL"]
        + ["*SET_NODE_LIST", "*SET_PART_LIST", "*SET_SHELL_LIST"]
    },
}
TopoClsMap = {
    "nodes": ["*NODE"],
    "elems": ["*ELEMENT_SOLID", "*ELEMENT_SHELL", "*ELEMENT_SHELL_THICKNESS"]
    + ["*ELEMENT_BEAM", "*ELEMENT_BEAM_OFFSET", "*ELEMENT_BEAM_ORIENTATION"],
    "parts": ["*PART"],
    "define_curve": ["*DEFINE_CURVE"],
    "set_list": ["*SET_NODE", "*SET_PART", "*SET_SHELL"]
    + ["*SET_NODE_LIST", "*SET_PART_LIST", "*SET_SHELL_LIST"]
    + ["*SET_NODE_TITLE", "*SET_PART_TITLE", "*SET_SHELL_TITLE"]
    + ["*SET_NODE_LIST_TITLE", "*SET_PART_LIST_TITLE", "*SET_SHELL_LIST_TITLE"],
}


def reshape_list(ids: list = [0], _f_W: int = 10, _m_l: int = 80):
    _f_n = int(_m_l / _f_W)
    res = []
    for i in range(int(np.ceil(len(ids) / (_f_n)))):
        res.append([each for each in ids[i * _f_n : (i + 1) * _f_n]])
    return res


def convert_to_tuple(nested_list):
    return nested_list


def split_bywidth(line, widths: list[int], replace_param: dict[str, str] = None):
    line = line.rstrip("\n")
    fields = []
    index = 0
    while line and index < len(widths):
        fields.append(line[0 : widths[index]])
        line = line[widths[index] :]
        index += 1
    if replace_param:
        _fields = []
        for s in fields:
            _p_s = s.strip()
            if _p_s[0] == "&" and _p_s[1:] in replace_param.keys():
                s = replace_param[_p_s[1:]]
            _fields.append(s)
        fields = _fields
    return fields


def format_numeric2str(value: int | float, len_fomrat: int = 8):
    int_part = dec_part = ""
    len_fomrat = len_fomrat if len_fomrat > 7 else 7
    if isinstance(value, float):
        _strofvalue = f"{value:.{len_fomrat}f}".split(".", 1)
        int_part = _strofvalue[0]
        dec_part = _strofvalue[1].rstrip("0") or "0"
    else:
        int_part = str(value)
    if int_part[0] in ["+", "-"]:
        int_part = int_part[1:]
    int_len = len(int_part)
    dec_len = len(dec_part)
    if int_len > len_fomrat - 1:
        if int_len == len_fomrat and value >= 0:
            result = f"{int(value):{len_fomrat}d}"
        else:
            result = f"{value:+.{len_fomrat-7}e}"
    elif 1 < int_len:
        result = f"{value:>.{dec_len}f}"[:len_fomrat]
    else:
        ix_isnot0 = (
            [i for i, ch in enumerate(dec_part) if ch != "0"][0] if dec_part not in ["", "0"] else 0
        )
        if ix_isnot0 > len_fomrat - 7 + (2 if value >= 0 else 1):
            result = f"{value:.{len_fomrat-7}e}"
        else:
            result = f"{value:>.{dec_len}f}"[:len_fomrat]
    result = result.rjust(len_fomrat) if result else "不能格式化"
    return result


def split_sequence(seq, num):
    base_length = len(seq) // num
    remainder = len(seq) % num
    result = []
    start_idx = 0
    for i in range(num):
        current_length = base_length + (1 if i < remainder else 0)
        end_idx = start_idx + current_length
        result.append(seq[start_idx:end_idx])
        start_idx = end_idx
    return result


class __LsDyna_Base:
    def __init__(
        self,
        outer_obj,
        keyword: str = "",
        cards: list[str] = [""],
        keyword_settings: str = "",
    ):
        self.__dict__["__is_init__"] = True
        self.__outer_obj__: bl_keyfile = outer_obj
        self.keyword: str = keyword.upper()
        self.keyword_settings: str = keyword_settings
        self.cards: list[str] = cards
        self.__set_onlyin_inner__ = [
            "__outer_obj__",
            "keyword",
            "keyword_settings",
            "__set_str__",
            "str",
            "str_cardsonly",
            "card_EX",
            "__str_cardsonly__",
            "__set_onlyin_inner__",
        ]
        self.__dict__["__is_init__"] = False

    def __set_str__(self):
        self.__dict__["__is_inner__"] = True
        str_title = "".join(
            [
                f"{self.keyword}",
                ((" " + self.keyword_settings) if self.keyword_settings else ""),
                "\n",
            ]
        )
        self.str_cardsonly = "".join(self.cards)
        self.str = str_title + self.str_cardsonly
        self.__dict__["__is_inner__"] = False

    def __str__(self) -> str:
        self.__dict__["__is_inner__"] = True
        self.__set_str__()
        self.__dict__["__is_inner__"] = False
        return self.str

    def __setattr__(self, ww, value):
        if self.__is_init__:
            self.__dict__[ww] = value
        elif self.__dict__.get("__is_inner__", False):
            self.__dict__[ww] = value
        else:
            if ww not in self.__set_onlyin_inner__:
                self.__dict__[ww] = value
                self.__set_str__()
            else:
                print(f"属性{ww}是只读属性 不能被修改")
            if self.is_edited or (ww in ["__reset__"]):
                self.__outer_obj__._bl_keyfile__diff_kf["mod"].append(
                    [self.keyword, self.str_cardsonly]
                )
                if self.keyword in sum(self.__outer_obj__.__topocls_name__.values(), []):
                    _pd_newkw = self.__outer_obj__.__update_kwdf__(self)
                    _pd_all = self.__outer_obj__.keywords[self.keyword]
                    _pd_all.loc[_pd_all.obj == self, :] = _pd_newkw
                    self.__outer_obj__._bl_keyfile__filtercache = {}

    def __deepcopy__(self, memo):
        self.__dict__["__is_inner__"] = True
        new_obj = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            if k == "__outer_obj__":
                new_obj.__dict__[k] = self.__outer_obj__
            else:
                new_obj.__dict__[k] = copy.deepcopy(v, memo)
        return new_obj

    @property
    def is_edited(self):
        self.__dict__["__is_inner__"] = True
        ...
        self.__dict__["__is_inner__"] = False
        return self.str_cardsonly != self.__str_cardsonly__

    def save(self, filename, with_title=True, permission="w"):
        self.__dict__["__is_inner__"] = True
        write_str = self.str if with_title else self.str_cardsonly
        with open(filename, permission) as f:
            f.write(write_str)
        self.__dict__["__is_inner__"] = False


class LsDyna_ENTITY(__LsDyna_Base):
    def __init__(
        self,
        outer_obj,
        keyword: str = "",
        cards: list[str] = [""],
        keyword_settings: str = "",
        cardfield=0,
        pagmfield=0,
    ):
        super().__init__(
            outer_obj=outer_obj,
            keyword=keyword,
            cards=cards,
            keyword_settings=keyword_settings,
        )
        self.__dict__["__is_init__"] = True
        self.__set_onlyin_inner__ += ["_cardfield", "_pagmfield"]
        self.__set_additional_info(cardfield, pagmfield)
        self.__set_str__()
        self.__str_cardsonly__ = self.str_cardsonly
        self.__dict__["__is_init__"] = False

    def __set_additional_info(self, cardfield: list, pagmfield: dict):
        self.__dict__["__is_inner__"] = True
        keyword = self.keyword
        if cardfield:
            _cardfield = cardfield
        else:
            _ec = self.__outer_obj__._bl_keyfile__EntityCls_CardFields
            _ec_k = _ec.keys()
            if keyword in _ec_k:
                _cardfield = _ec[keyword]
            else:
                _keys = sorted([x for x in _ec_k if keyword.startswith(x)], key=len, reverse=True)
                _cardfield = _ec[_keys[0]] if _keys else [[]]
        if pagmfield:
            _pagmfield = pagmfield
        else:
            _ep = self.__outer_obj__._bl_keyfile__EntityCls_PagmFields
            _ep_k = _ep.keys()
            if keyword in _ep_k:
                _pagmfield = _ep[keyword]
            else:
                _keys = sorted([x for x in _ep_k if keyword.startswith(x)], key=len, reverse=True)
                _pagmfield = _ep[_keys[0]] if _keys else {}
        _ck = [0, 0]
        for v in _pagmfield.values():
            _sst = v["index"][0]
            if isinstance(_sst, str) and ":" in _sst:
                try:
                    _sst_split = _sst.split(":")
                    _slc = slice(*map(lambda x: int(x) if x else None, _sst_split))
                    _r_c = list(range(*_slc.indices(len(self.cards))))
                    if _r_c:
                        if len(_sst_split) == 2:
                            _cardfield = _cardfield[: _r_c[0]] + _cardfield[
                                _r_c[0] : _r_c[0] + 1
                            ] * len(_r_c)
                            _ck[0] += 1
                        elif len(_sst_split) == 3:
                            _cc = [
                                _cardfield[i] if i < len(_cardfield) else []
                                for i in range(_r_c[-1] + 1)
                            ] + _cardfield[_r_c[-1] + 1 :]
                            for i in _r_c:
                                _cc[i] = _cardfield[_r_c[0]]
                            _cardfield = _cc
                            _ck[1] += 1
                except:
                    print(self.keyword, "\n", _sst, "\n", _r_c, "\n", _cc)
                    raise ValueError("展开不定长卡片索引错误")
        if all(_ck):
            raise ValueError("关键字的不定长行卡片只能为同一种形态")
        self._cardfield = convert_to_tuple(_cardfield)
        self._pagmfield = _pagmfield
        self.__dict__["__is_inner__"] = False

    def __getitem__(self, pos):
        def __get_card_field(card, field):
            try:
                _card_field = self._cardfield[card]
                _left = sum(_card_field[:field]) if field else 0
                _right = sum(_card_field[: field + 1])
                return self.cards[card][_left:_right]
            except:
                return "无法以指定索引获取字段"

        self.__dict__["__is_inner__"] = True
        range_card, range_field = [], []
        if isinstance(pos, tuple):
            if isinstance(pos[0], int):
                range_card = [pos[0]]
            elif isinstance(pos[0], slice):
                range_card = list(range(*pos[0].indices(len(self._cardfield))))
            if isinstance(pos[1], int):
                range_field = [[pos[1]]]
            elif isinstance(pos[1], slice):
                range_field = [
                    list(range(*pos[1].indices(len(e))))
                    for e in [self._cardfield[i] for i in range_card]
                ]
        elif isinstance(pos, slice):
            range_card = list(range(*pos.indices(len(self._cardfield))))
            range_field = [list(range(len(e))) for e in [self._cardfield[i] for i in range_card]]
        elif isinstance(pos, str):
            if pos in self._pagmfield.keys():
                card, field = self._pagmfield[pos]["index"]
                if any(isinstance(x, str) for x in [card, field]):
                    if isinstance(card, str):
                        _slc = slice(*map(lambda x: int(x) if x else None, card.split(":")))
                        range_card = list(range(*_slc.indices(len(self.cards))))
                    else:
                        raise TypeError("错误的索引方法")
                    if isinstance(field, str):
                        _slc = slice(*map(lambda x: int(x) if x else None, field.split(":")))
                        range_field = [
                            list(range(*_slc.indices(len(e))))
                            for e in [self._cardfield[i] for i in range_card]
                        ]
                    elif isinstance(field, int):
                        range_field = [[field]] * len(range_card)
                    else:
                        raise TypeError("错误的索引方法")
                else:
                    range_card = [card]
                    range_field = [[field]]
            else:
                raise KeyError(f"字段 '{pos}' 不在_pagmfield中")
        elif isinstance(pos, int):
            range_card = [pos]
            range_field = [list(range(len(self._cardfield[pos])))]
        else:
            raise TypeError("错误的索引方法")
        picks = []
        if not (range_card and range_field):
            ...
        else:
            _ep = [[_c, _f] for _c, _fs in zip(range_card, range_field) for _f in _fs]
            _ep = [
                x
                for x in _ep
                if len(self.cards[x[0]].rstrip("\n")) >= sum(self._cardfield[x[0]][: x[1] + 1])
            ]
            _pick_d = defaultdict(list)
            for _i_c, _i_f in _ep:
                _pick_str = __get_card_field(_i_c, _i_f)
                if _pick_str != []:
                    _pick_d[_i_c].append(_pick_str)
            picks = [[f"line_{k}", v] for k, v in _pick_d.items() if v]
        _l_p = len(picks)
        if _l_p > 1:
            ...
        elif _l_p == 1:
            if len(picks[0][1]) > 1:
                picks = picks[0][1]
            else:
                picks = picks[0][1][0]
        else:
            picks = "索引错误 或 未编码全部卡片索引"
        self.__dict__["__is_inner__"] = False
        return picks

    def __setitem__(self, pos, value):
        def __set_card_field(card, field, value):
            try:
                _card_field = self._cardfield[card]
                if isinstance(value, (int, float)):
                    value = format_numeric2str(value, _card_field[field])
                elif isinstance(value, str):
                    value = (
                        f"{value:<{_card_field[field]}s}"
                        if len(value) < _card_field[field]
                        else value[: _card_field[field]]
                    )
                else:
                    raise ValueError("赋值类型错误")
                _left = sum(_card_field[:field]) if field else 0
                _right = sum(_card_field[: field + 1])
                self.cards[card] = self.cards[card][:_left] + value + self.cards[card][_right:]
                if self.is_edited:
                    self.__outer_obj__._bl_keyfile__diff_kf["mod"].append(
                        [self.keyword, self.str_cardsonly]
                    )
                    if self.keyword in sum(self.__outer_obj__.__topocls_name__.values(), []):
                        _pd_newkw = self.__outer_obj__.__update_kwdf__(self)
                        _pd_all = self.__outer_obj__.keywords[self.keyword]
                        _pd_all.loc[_pd_all.obj == self, :] = _pd_newkw
                        self.__outer_obj__._bl_keyfile__filtercache = {}
                return self.cards[card]
            except:
                return "索引错误 或 未编码全部卡片索引"

        self.__dict__["__is_inner__"] = True
        _excl_kw = sum(self.__outer_obj__.__topocls_name__.values(), [])
        if not self.keyword in _excl_kw:
            range_card, range_field = [], []
            if isinstance(pos, tuple):
                if all([isinstance(each, int) for each in pos]):
                    range_card = [pos[0]]
                    range_field = [[pos[1]]]
                else:
                    raise ValueError("位置索引复制必须是 [int,int] 或 [int]")
            elif isinstance(pos, slice):
                raise ValueError("位置索引复制必须是 [int,int] 或 [int]")
            elif isinstance(pos, str):
                if pos in self._pagmfield.keys():
                    card, field = self._pagmfield[pos]["index"]
                    if any(isinstance(x, str) for x in [card, field]):
                        if isinstance(card, str):
                            _slc = slice(*map(lambda x: int(x) if x else None, card.split(":")))
                            range_card = list(range(*_slc.indices(len(self.cards))))
                        else:
                            raise TypeError("错误的索引方法")
                        if isinstance(field, str):
                            _slc = slice(*map(lambda x: int(x) if x else None, field.split(":")))
                            range_field = [
                                list(range(*_slc.indices(len(e))))
                                for e in [self._cardfield[i] for i in range_card]
                            ]
                        elif isinstance(field, int):
                            range_field = [[field]] * len(range_card)
                        else:
                            raise TypeError("错误的索引方法")
                    else:
                        range_card = [card]
                        range_field = [[field]]
                else:
                    raise KeyError(f"字段 '{pos}' 不在_pagmfield中")
            elif isinstance(pos, int):
                range_card = [pos]
                range_field = "FullLine"
            else:
                raise TypeError("错误的索引方法")
            result = ""
            if not (range_card and range_field):
                result = "索引错误 或 未编码全部卡片索引"
            else:
                if range_field == "FullLine":
                    self.cards[range_card[0]] = value
                else:
                    _ep = [[_c, _f] for _c, _fs in zip(range_card, range_field) for _f in _fs]
                    _ep = [
                        x
                        for x in _ep
                        if len(self.cards[x[0]].rstrip("\n"))
                        >= sum(self._cardfield[x[0]][: x[1] + 1])
                    ]
                    _l_s = len(_ep)
                    if _l_s == 1:
                        if not isinstance(value, (list, tuple)):
                            value = [value]
                    else:
                        if len(value) != _l_s:
                            raise TypeError("目标和字段数量不相等")
                    for i, (c, f) in enumerate(_ep):
                        __set_card_field(c, f, value[i])
                self.__set_str__()
                result = self.cards
        else:
            result = f"不处理{_excl_kw}"
        print(result)
        self.__dict__["__is_inner__"] = False

    def __repr__(self):
        self.__dict__["__is_inner__"] = True
        _c_f = self._cardfield
        _p_f = self._pagmfield
        repr_str = "".join(
            [
                "".join([f"{x}".ljust(10, "-") for x in range(8)]),
                "\n",
                "-" * 80,
                "\n",
                self.str,
                "-" * 80,
                "\n",
                f"_cardfield:{len(_c_f)}",
                "\n",
                "".join(
                    [
                        f"  |_ {n}:{l}\n"
                        for n, l in [(len(list(group)), key) for key, group in groupby(_c_f)]
                    ]
                ),
                f"_pagmfield:{_p_f.keys()}",
                "\n",
                "".join([f"  |_ {k}:{v}\n" for k, v in _p_f.items()]),
            ]
        )
        self.__dict__["__is_inner__"] = False
        return repr_str

    def reset(self):
        self.__dict__["__is_inner__"] = True
        self.__init__(
            outer_obj=self.__outer_obj__,
            **self.__outer_obj__.__read_kwstr__(
                kf_lines=[self.keyword + " " + self.keyword_settings]
                + [x + "\n" for x in self.__str_cardsonly__.split("\n")[:-1]],
                only_pre=1,
            ),
        )
        self.__dict__["__is_inner__"] = False
        self.__reset__ = True


class LsDyna_NODE(__LsDyna_Base):
    def __init__(
        self,
        outer_obj,
        id,
        x,
        y,
        z,
        card1_add_fields: dict[str, int] = {"TC": "", "RC": ""},
        card_EX="",
        keyword="*NODE",
        keyword_settings: str = "",
    ):
        super().__init__(
            outer_obj=outer_obj,
            keyword=keyword,
            cards=[""],
            keyword_settings=keyword_settings,
        )
        self.__dict__["__is_init__"] = True
        self.__set_onlyin_inner__ += ["_cardfield", "coords"]
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        card1_add_fields = {
            **{"TC": "", "RC": ""},
            **card1_add_fields,
        }
        self.card1_add_fields = [
            float(each) if not isinstance(each, str) else "" for each in card1_add_fields.values()
        ]
        self.card_EX = card_EX
        self._cardfield = convert_to_tuple(
            self.__outer_obj__._bl_keyfile__EntityCls_CardFields[self.keyword]
        )
        self.__set_str__()
        self.__str_cardsonly__ = self.str_cardsonly
        self.__dict__["__is_init__"] = False

    def get_related_elems(self, return_asdf=0):
        self.__dict__["__is_inner__"] = True
        related_elems = self.__outer_obj__.filter_TopoDF_by_ids(
            et_type="elems", field="id_nodes", ids=[self.id], return_asdf=return_asdf
        )
        self.__dict__["__is_inner__"] = False
        return related_elems

    def __set_str__(self):
        self.__dict__["__is_inner__"] = True
        str_title = "".join(
            [
                f"{self.keyword}",
                ((" " + self.keyword_settings) if self.keyword_settings else ""),
                "\n",
            ]
        )
        str_field_comments = (
            "$    nid               x               y               z      tc      rc\n".replace(
                " ", "-"
            )
        )
        _cf = self._cardfield
        _cf_0 = _cf[0]
        _fn2s = format_numeric2str
        str_cardsonly_parts = [
            _fn2s(self.id, _cf_0[0]),
            "".join(
                [
                    _fn2s(each, _cf_0[index])
                    for index, each in ((1, self.x), (2, self.y), (3, self.z))
                ]
            ),
            "".join(
                [
                    (_fn2s(each, _cf_0[index + 4]) if not each == "" else " " * _cf_0[index + 4])
                    for index, each in enumerate(self.card1_add_fields)
                ]
            ),
            "\n",
        ]
        self.str_cardsonly = "".join(str_cardsonly_parts) + self.card_EX
        self.str = str_title + str_field_comments + self.str_cardsonly
        self.cards = [s + "\n" for s in self.str_cardsonly.split("\n") if s]
        self.__dict__["__is_inner__"] = False

    def __repr__(self):
        self.__dict__["__is_inner__"] = True
        lines = [f"node with:"]
        lines.extend(
            [
                f"  ID:".ljust(10) + f"{self.id}",
                f"  Coords:".ljust(10) + f"{(self.x, self.y, self.z)}",
            ]
        )
        _r_e = self.get_related_elems(1)
        if _r_e:
            lines.append(f"  Elems:".ljust(10) + f"{sum([len(x) for x in _r_e])}")
            lines.append(f"    ids:".ljust(10) + f"{[x.id.tolist() for x in _r_e]}")
        else:
            lines.append(f"  Elems:".ljust(10) + f"None, is a isolated node")
        self.__dict__["__is_inner__"] = False
        return "\n".join(lines)

    def reset(self):
        self.__dict__["__is_inner__"] = True
        self.__init__(
            outer_obj=self.__outer_obj__,
            **self.__outer_obj__.get_nodes(self.__str_cardsonly__).iloc[0, :].to_dict(),
        )
        self.__dict__["__is_inner__"] = False
        self.__reset__ = True


class __LsDyna_Elem_Factory(__LsDyna_Base):
    def __init__(self, outer_obj, keyword, id, id_part, id_nodes, card_EX, keyword_settings=""):
        super().__init__(
            outer_obj=outer_obj,
            keyword="*ELEMENT_*",
            cards=[""],
            keyword_settings=keyword_settings,
        )
        self.__dict__["__is_init__"] = True
        self.keyword = keyword
        self.id = int(id)
        self.id_part = int(id_part)
        self.id_nodes = [int(i) for i in id_nodes]
        self.reshape_nodes(id_nodes)
        self.card_EX = card_EX
        self.__dict__["__is_init__"] = False

    def reshape_nodes(self, id_nodes):
        self.__dict__["__is_inner__"] = True
        _l_in = len(self.id_nodes)
        self.__id_nodes__ = self.id_nodes
        if "SOLID" in self.keyword:
            if _l_in < 6:
                self.__id_nodes__ = self.id_nodes + [self.id_nodes[-1]] * (8 - _l_in)
            elif _l_in == 6:
                self.__id_nodes__ = (
                    self.id_nodes[:4] + [self.id_nodes[4]] * 2 + [self.id_nodes[5]] * 2
                )
        self.__dict__["__is_inner__"] = False

    def get_related_nodes(self, return_asdf=0):
        self.__dict__["__is_inner__"] = True
        related_nodes = self.__outer_obj__.filter_TopoDF_by_ids(
            et_type="nodes", field="id", ids=self.id_nodes, return_asdf=return_asdf
        )
        self.__dict__["__is_inner__"] = False
        return related_nodes[0] if related_nodes else None

    def get_centercoords(self, return_size=0):
        self.__dict__["__is_inner__"] = True
        _rns = self.get_related_nodes()
        _rnd_c = [(n["x"], n["y"], n["z"]) for n in _rns]
        _len = len(_rnd_c)
        _center = (
            round(sum([n[0] for n in _rnd_c]) / _len, 5),
            round(sum([n[1] for n in _rnd_c]) / _len, 5),
            round(sum([n[2] for n in _rnd_c]) / _len, 5),
        )
        _dist = "平均边长未计算"
        if return_size:
            _rnd_c = _rnd_c[:5]
            dist_matrix = distance.cdist(_rnd_c, _rnd_c, "euclidean")
            _dist = np.mean(dist_matrix[dist_matrix != 0])
        self.__dict__["__is_inner__"] = False
        return _center, _dist

    def get_related_part(self, return_asdf=0):
        self.__dict__["__is_inner__"] = True
        related_part = self.__outer_obj__.filter_TopoDF_by_ids(
            et_type="parts", field="id", ids=[self.id_part], return_asdf=return_asdf
        )
        self.__dict__["__is_inner__"] = False
        return related_part[0] if related_part else None

    def __repr__(self) -> str:
        self.__dict__["__is_inner__"] = True
        lines = [f"{self.keyword} with:"]
        lines.append(f"  ID:".ljust(10) + f"{self.id}")
        centercoords = self.get_centercoords(1)
        if centercoords:
            lines.append(f"  Coords:".ljust(10) + f"{centercoords}")
        lines.append(f"  Nodes:".ljust(10) + f"{len(self.__id_nodes__)}")
        lines.append(f"    ids:".ljust(10) + f"{self.__id_nodes__}")
        lines.append(f"  partid:".ljust(10) + f"{self.id_part}")
        self.__dict__["__is_inner__"] = False
        return "\n".join(lines)

    def reset(self):
        self.__dict__["__is_inner__"] = True
        self.__init__(
            outer_obj=self.__outer_obj__,
            **self.__outer_obj__.get_elems(self.__str_cardsonly__, kw_type=self.keyword)
            .iloc[0, :]
            .to_dict(),
        )
        self.__dict__["__is_inner__"] = False
        self.__reset__ = True


class LsDyna_ELEMENT_SOLID(__LsDyna_Elem_Factory):
    def __init__(
        self,
        outer_obj,
        id,
        id_part,
        id_nodes,
        keyword="*ELEMENT_SOLID",
        card_EX="",
        keyword_settings="",
    ):
        super().__init__(
            outer_obj,
            keyword=keyword,
            id=id,
            id_part=id_part,
            id_nodes=id_nodes,
            card_EX=card_EX,
            keyword_settings=keyword_settings,
        )
        self.__dict__["__is_init__"] = True
        self._cardfield = convert_to_tuple(
            self.__outer_obj__._bl_keyfile__EntityCls_CardFields[self.keyword]
        )
        self.__set_str__()
        self.__str_cardsonly__ = self.str_cardsonly
        self.__dict__["__is_init__"] = False

    def __set_str__(self):
        self.__dict__["__is_inner__"] = True
        str_title = "".join(
            [
                f"{self.keyword}",
                ((" " + self.keyword_settings) if self.keyword_settings else ""),
                "\n",
            ]
        )
        str_field_comments = (
            "$    EID     PID\n"
            + "$     N1      N2      N3      N4      N5      N6      N7      N8\n"
        ).replace(" ", "-")
        _cf = self._cardfield
        _cf_0 = _cf[0]
        _cf_1 = _cf[1]
        _fn2s = format_numeric2str
        str_cardsonly_parts = [
            _fn2s(self.id, _cf_0[0]),
            _fn2s(self.id_part, _cf_0[1]),
            "\n",
            "".join([_fn2s(each, _cf_1[index]) for index, each in enumerate(self.__id_nodes__)]),
            "\n",
        ]
        self.str_cardsonly = "".join(str_cardsonly_parts) + self.card_EX
        self.str = str_title + str_field_comments + self.str_cardsonly
        self.cards = [s + "\n" for s in self.str_cardsonly.split("\n") if s]
        self.__dict__["__is_inner__"] = False


class LsDyna_ELEMENT_SHELL(__LsDyna_Elem_Factory):
    def __init__(
        self,
        outer_obj,
        id,
        id_part,
        id_nodes,
        keyword="*ELEMENT_SHELL",
        card_EX="",
        keyword_settings="",
    ):
        super().__init__(
            outer_obj,
            keyword=keyword,
            id=id,
            id_part=id_part,
            id_nodes=id_nodes,
            card_EX=card_EX,
            keyword_settings=keyword_settings,
        )
        self.__dict__["__is_init__"] = True
        self._cardfield = convert_to_tuple(
            self.__outer_obj__._bl_keyfile__EntityCls_CardFields[self.keyword]
        )
        self.__set_str__()
        self.__str_cardsonly__ = self.str_cardsonly
        self.__dict__["__is_init__"] = False

    def __set_str__(self):
        self.__dict__["__is_inner__"] = True
        str_title = "".join(
            [
                f"{self.keyword}",
                ((" " + self.keyword_settings) if self.keyword_settings else ""),
                "\n",
            ]
        )
        str_field_comments = "$    EID     PID      N1      N2      N3      N4\n".replace(" ", "-")
        _cf = self._cardfield
        _cf_0 = _cf[0]
        _fn2s = format_numeric2str
        str_cardsonly_parts = [
            _fn2s(self.id, _cf_0[0]),
            _fn2s(self.id_part, _cf_0[1]),
            "".join(
                [_fn2s(each, _cf_0[index + 2]) for index, each in enumerate(self.__id_nodes__)]
            ),
            "\n",
        ]
        self.str_cardsonly = "".join(str_cardsonly_parts) + self.card_EX
        self.str = str_title + str_field_comments + self.str_cardsonly
        self.cards = [s + "\n" for s in self.str_cardsonly.split("\n") if s]
        self.__dict__["__is_inner__"] = False


class LsDyna_ELEMENT_BEAM(__LsDyna_Elem_Factory):
    def __init__(
        self,
        outer_obj,
        id: int,
        id_part: int,
        id_nodes: list[int],
        card1_add_fields: dict[str, int] = {
            "N3": "",
            "RT1": "",
            "RR1": "",
            "RT2": "",
            "RR2": "",
            "LOCAL": "",
        },
        keyword="*ELEMENT_BEAM",
        card_EX="",
        keyword_settings: str = "",
    ):
        super().__init__(
            outer_obj,
            keyword=keyword,
            id=id,
            id_part=id_part,
            id_nodes=id_nodes,
            card_EX=card_EX,
            keyword_settings=keyword_settings,
        )
        self.__dict__["__is_init__"] = True
        card1_add_fields = {
            **{
                "N3": "",
                "RT1": "",
                "RR1": "",
                "RT2": "",
                "RR2": "",
                "LOCAL": "",
            },
            **card1_add_fields,
        }
        self.card1_add_fields = [
            int(each) if not isinstance(each, str) else "" for each in card1_add_fields.values()
        ]
        self._cardfield = convert_to_tuple(
            self.__outer_obj__._bl_keyfile__EntityCls_CardFields[self.keyword]
        )
        self.__set_str__()
        self.__str_cardsonly__ = self.str_cardsonly
        self.__dict__["__is_init__"] = False

    def __set_str__(self):
        self.__dict__["__is_inner__"] = True
        str_title = "".join(
            [
                f"{self.keyword}",
                ((" " + self.keyword_settings) if self.keyword_settings else ""),
                "\n",
            ]
        )
        str_field_comments = "$     ID     PID      N1      N2      N3     RT1     RR1     RT2     RR2   LOCAL\n".replace(
            " ", "-"
        )
        _cf = self._cardfield
        _cf_0 = _cf[0]
        _fn2s = format_numeric2str
        str_cardsonly_parts = [
            _fn2s(self.id, _cf_0[0]),
            _fn2s(self.id_part, _cf_0[1]),
            "".join(
                [
                    (_fn2s(each, _cf_0[index + 2]) if not each == "" else " " * _cf_0[index + 2])
                    for index, each in enumerate(self.__id_nodes__ + self.card1_add_fields)
                ]
            ),
            "\n",
        ]
        self.str_cardsonly = "".join(str_cardsonly_parts) + self.card_EX
        self.str = str_title + str_field_comments + self.str_cardsonly
        self.cards = [s + "\n" for s in self.str_cardsonly.split("\n") if s]
        self.__dict__["__is_inner__"] = False


class LsDyna_PART(__LsDyna_Base):
    def __init__(
        self,
        outer_obj,
        name: str,
        id: int,
        id_sec: int,
        id_mat: int,
        card2_add_fields: dict[str, int] = {
            "EOSID": "",
            "HGID": "",
            "GRAV": "",
            "ADPOPT": "",
            "TMID": "",
        },
        card_EX="",
        keyword="*PART",
        keyword_settings: str = "",
    ):
        super().__init__(
            outer_obj=outer_obj,
            keyword=keyword,
            cards=[""],
            keyword_settings=keyword_settings,
        )
        self.__dict__["__is_init__"] = True
        self.name = str(name)
        self.id = int(id)
        self.id_sec = int(id_sec)
        self.id_mat = int(id_mat)
        card2_add_fields = {
            **{"EOSID": "", "HGID": "", "GRAV": "", "ADPOPT": "", "TMID": ""},
            **card2_add_fields,
        }
        self.card2_add_fields = [
            int(each) if not isinstance(each, str) else "" for each in card2_add_fields.values()
        ]
        self.card_EX = card_EX
        self._cardfield = convert_to_tuple(
            self.__outer_obj__._bl_keyfile__EntityCls_CardFields["*PART"]
        )
        self.__set_str__()
        self.__str_cardsonly__ = self.str_cardsonly
        self.__dict__["__is_init__"] = False

    def get_related_elems(self, return_asdf=0):
        self.__dict__["__is_inner__"] = True
        related_elems = self.__outer_obj__.filter_TopoDF_by_ids(
            et_type="elems", field="id_part", ids=[self.id], return_asdf=return_asdf
        )
        self.__dict__["__is_inner__"] = False
        return related_elems

    def get_related_mat(self, return_asdf=0):
        self.__dict__["__is_inner__"] = True
        related_mat = self.__outer_obj__.filter_TopoDF_by_ids(
            et_type="mats", field="id", ids=[self.id_mat], return_asdf=return_asdf
        )
        self.__dict__["__is_inner__"] = False
        return related_mat[0] if related_mat else None

    def get_related_section(self, return_asdf=0):
        self.__dict__["__is_inner__"] = True
        related_section = self.__outer_obj__.filter_TopoDF_by_ids(
            et_type="sections", field="id", ids=[self.id_sec], return_asdf=return_asdf
        )
        self.__dict__["__is_inner__"] = False
        return related_section[0] if related_section else None

    def __set_str__(self):
        self.__dict__["__is_inner__"] = True
        str_title = "".join(
            [
                f"{self.keyword}",
                ((" " + self.keyword_settings) if self.keyword_settings else ""),
                "\n",
            ]
        )
        str_field_comments = (
            "$NAME".ljust(80)
            + "\n"
            + "$      PID     SECID       MID     EOSID      HGID      GRAV    ADPOPT      TMID\n"
        ).replace(" ", "-")
        _cf = self._cardfield
        _cf_0 = _cf[0]
        _cf_1 = _cf[1]
        _fn2s = format_numeric2str
        str_cardsonly_parts = [
            (f"{self.name:<{_cf_0[0]}s}" if len(self.name) < _cf_0[0] else self.name[: _cf_0[0]]),
            "\n",
            "".join(
                [
                    _fn2s(each, _cf_1[index])
                    for index, each in enumerate([self.id, self.id_sec, self.id_mat])
                ]
            ),
            "".join(
                [
                    (_fn2s(each, _cf_1[index + 3]) if not each == "" else " " * _cf_1[index + 3])
                    for index, each in enumerate(self.card2_add_fields)
                ]
            ),
            "\n",
        ]
        self.str_cardsonly = "".join(str_cardsonly_parts) + self.card_EX
        self.str = str_title + str_field_comments + self.str_cardsonly
        self.cards = [s + "\n" for s in self.str_cardsonly.split("\n") if s]
        self.__dict__["__is_inner__"] = False

    def __repr__(self):
        self.__dict__["__is_inner__"] = True
        lines = [f"part with:"]
        lines.extend(
            [
                f"  NAME:".ljust(10) + f"{self.name}",
                f"  ID:".ljust(10) + f"{self.id}",
                f"  ID_SEC:".ljust(10) + f"{self.id_sec}",
                f"  ID_MAT:".ljust(10) + f"{self.id_mat}",
            ]
        )
        _r_e = self.get_related_elems(1)
        if _r_e:
            lines.append(f"  Elems:".ljust(10) + f"{sum([len(x) for x in _r_e])}")
            lines.append(f"    ids:".ljust(10) + f"{[x.id.tolist() for x in _r_e]}")
        else:
            lines.append(f"  Elems:".ljust(10) + f"None, is a isolated part")
        self.__dict__["__is_inner__"] = False
        return "\n".join(lines)

    def reset(self):
        self.__dict__["__is_inner__"] = True
        self.__init__(
            outer_obj=self.__outer_obj__,
            **self.__outer_obj__.get_parts(self.__str_cardsonly__).iloc[0, :].to_dict(),
        )
        self.__dict__["__is_inner__"] = False
        self.__reset__ = True


class LsDyna_DEFINE_CURVE(__LsDyna_Base):
    def __init__(
        self,
        outer_obj,
        id,
        x,
        y,
        sidr="",
        sfa=1.0,
        sfo=1.0,
        offa="",
        offo="",
        dattyp="",
        lcint="",
        card_EX="",
        keyword="*DEFINE_CURVE",
        keyword_settings="",
    ):
        super().__init__(
            outer_obj=outer_obj,
            keyword=keyword,
            cards=[""],
            keyword_settings=keyword_settings,
        )
        self.__dict__["__is_init__"] = True
        self.id = int(id)
        self.x = x
        self.y = y
        self.sidr = int(sidr) if not sidr == "" else ""
        self.sfa = float(sfa) if not sfa == "" else ""
        self.sfo = float(sfo) if not sfo == "" else ""
        self.offa = float(offa) if not offa == "" else ""
        self.offo = float(offo) if not offo == "" else ""
        self.dattyp = int(dattyp) if not dattyp == "" else ""
        self.lcint = int(lcint) if not lcint == "" else ""
        self.card_EX = card_EX
        self._cardfield = convert_to_tuple(
            self.__outer_obj__._bl_keyfile__EntityCls_CardFields["*DEFINE_CURVE"]
        )
        self.__set_str__()
        self.__str_cardsonly__ = self.str_cardsonly
        self.__dict__["__is_init__"] = False

    def __set_str__(self):
        self.__dict__["__is_inner__"] = True
        str_title = "".join(
            [
                f"{self.keyword}",
                ((" " + self.keyword_settings) if self.keyword_settings else ""),
                "\n",
            ]
        )
        str_field_comments = (
            "$     LCID      SIDR       SFA       SFO      OFFA      OFFO    DATTYP     LCINT"
            + "\n"
            + "$                  X                   Y\n"
        ).replace(" ", "-")
        _cf = self._cardfield
        _cf_0 = _cf[0]
        _cf_1 = _cf[1]
        _fn2s = format_numeric2str
        str_cardsonly_parts = [
            "".join(
                [
                    _fn2s(each, _cf_0[index]) if not each == "" else " " * _cf_0[index]
                    for index, each in enumerate(
                        [
                            self.id,
                            self.sidr,
                            self.sfa,
                            self.sfo,
                            self.offa,
                            self.offo,
                            self.dattyp,
                            self.lcint,
                        ]
                    )
                ]
            ),
            "\n",
            "\n".join(
                [
                    (_fn2s(each[0], _cf_1[0]) + _fn2s(each[1], _cf_1[1]))
                    for each in zip(self.x, self.y)
                    if not any([x in [np.nan, []] for x in each])
                ]
            ),
            "\n",
        ]
        self.str_cardsonly = "".join(str_cardsonly_parts) + self.card_EX
        self.str = str_title + str_field_comments + self.str_cardsonly
        __cards = [s + "\n" for s in self.str_cardsonly.split("\n") if s]
        self.cards = [__cards[0], "".join(__cards[1:])]
        self.__dict__["__is_inner__"] = False

    def __repr__(self):
        self.__dict__["__is_inner__"] = True
        lines = [f"cruve with:"]
        lines.extend(
            [
                f"  ID:".ljust(10) + f"{self.id}",
                f"  XY:".ljust(10) + f"x:{len(self.x)} y:{len(self.y)}",
                f"    orix:".ljust(10) + f"{self.x}",
                f"    oriy:".ljust(10) + f"{self.y}",
                f"控制: 缩放 [{self.sfa} {self.sfo}] "
                + f"偏移 [{self.offa} {self.offa}] "
                + f"重插 {self.lcint} ",
            ]
        )
        self.__dict__["__is_inner__"] = False
        return "\n".join(lines)

    def reset(self):
        self.__dict__["__is_inner__"] = True
        self.__init__(
            outer_obj=self.__outer_obj__,
            **self.__outer_obj__.get_define_curve(self.__str_cardsonly__).iloc[0, :].to_dict(),
        )
        self.__dict__["__is_inner__"] = False
        self.__reset__ = True


class LsDyna_SET_LIST(__LsDyna_Base):
    def __init__(
        self,
        outer_obj,
        keyword,
        id: int,
        nids: list[list[int]],
        da1="",
        da2="",
        da3="",
        da4="",
        solver="MECH",
        card_EX="",
        keyword_settings="",
    ):
        super().__init__(
            outer_obj=outer_obj,
            keyword=keyword,
            cards=[""],
            keyword_settings=keyword_settings,
        )
        self.__dict__["__is_init__"] = True
        self.id = int(id)
        self.nids = [np.int32(i).tolist() for i in nids]
        self.da1 = float(da1) if not da1 == "" else ""
        self.da2 = float(da2) if not da2 == "" else ""
        self.da3 = float(da3) if not da3 == "" else ""
        self.da4 = float(da4) if not da4 == "" else ""
        self.solver = solver
        self.card_EX = card_EX
        self._cardfield = convert_to_tuple(
            self.__outer_obj__._bl_keyfile__EntityCls_CardFields[keyword]
        )
        self.__set_str__()
        self.__str_cardsonly__ = self.str_cardsonly
        self.__dict__["__is_init__"] = False

    def __set_str__(self):
        self.__dict__["__is_inner__"] = True
        str_title = "".join(
            [
                f"{self.keyword}",
                ((" " + self.keyword_settings) if self.keyword_settings else ""),
                "\n",
            ]
        )
        str_field_comments = (
            "$      SID       DA1       DA2       DA3       DA4    SOLVER\n"
            + "$     NID1      NID2      NID3      NID4      NID5      NID6      NID7      NID8\n"
        ).replace(" ", "-")
        _cf = self._cardfield
        _cf_0 = _cf[0]
        _cf_1 = _cf[1]
        _fn2s = format_numeric2str
        str_cardsonly_parts = [
            "".join(
                [
                    _fn2s(each, _cf_0[index]) if not each == "" else " " * _cf_0[index]
                    for index, each in enumerate(
                        [
                            self.id,
                            self.da1,
                            self.da2,
                            self.da3,
                            self.da4,
                        ]
                    )
                ]
                + [f"{self.solver:<{_cf_0[5]}s}"]
            ),
            "\n",
            "".join(
                [
                    _fn2s(each, _cf_1[index]) if not each == "" else " " * _cf_1[index]
                    for _line in self.nids
                    for index, each in enumerate(_line)
                ]
            ),
            "\n",
        ]
        self.str_cardsonly = "".join(str_cardsonly_parts) + self.card_EX
        self.str = str_title + str_field_comments + self.str_cardsonly
        __cards = [s + "\n" for s in self.str_cardsonly.split("\n") if s]
        self.cards = [__cards[0], "".join(__cards[1:])]
        self.__dict__["__is_inner__"] = False

    def __repr__(self):
        self.__dict__["__is_inner__"] = True
        lines = [f"set with:"]
        lines.extend(
            [
                f"  ID:".ljust(10) + f"{self.id}",
                f"  SOLVER:".ljust(10) + f"{self.solver}",
                f"  节点属性:".ljust(10)
                + "".join([f"{x} " for x in [self.da1, self.da2, self.da3, self.da4]]),
                f"  NIDS:".ljust(10) + f"{len(self.nids)}",
                f"    ids:".ljust(10) + f"{self.nids}",
            ]
        )
        self.__dict__["__is_inner__"] = False
        return "\n".join(lines)

    def reset(self):
        self.__dict__["__is_inner__"] = True
        self.__init__(
            outer_obj=self.__outer_obj__,
            **self.__outer_obj__.get_set_list(self.__str_cardsonly__, kw_type=self.keyword)
            .iloc[0, :]
            .to_dict(),
        )
        self.__dict__["__is_inner__"] = False
        self.__reset__ = True


class bl_keyfile:
    def __init__(
        self,
        keyfile,
        parsing_topo=True,
        is_init=1,
        acc_initbythread=0,
        encoding="utf-8",
    ):
        self.__set_params()
        self.__set_fieldconfig()
        if keyfile:
            self.kfilepath = pathlib.Path(keyfile)
            self.encoding = encoding
            if parsing_topo:
                self.__parsing_topo = parsing_topo
                self.__acc_initbythread = acc_initbythread
                self.__topocls_name__ = copy.deepcopy(TopoClsMap)
                self.read_kf(self.kfilepath)
                self.collect_PARAMETER()
                self.get_nodes(is_init=is_init)
                self.get_elems(is_init=is_init)
                self.get_parts(is_init=is_init)
                self.get_define_curve(is_init=is_init)
                self.get_set_list(is_init=is_init)
                self.collect_portion_MAT()
                self.collect_portion_SECTION()
            else:
                self.__topocls_name__ = {}
                self.read_kf(self.kfilepath)

    def __set_params(self):
        self.kfilepath = ""
        self.__parsing_topo = 0
        self.__acc_initbythread = 0
        self.acc_filterbycache = 1
        self.__filtercache = {}
        self.__topocls_name__ = {}
        self.__include_kw = (
            "*INCLUDE",
            "*INCLUDE_TRANSFORM",
            "*INCLUDE_PATH",
            "*INCLUDE_PATH_RELATIVE",
        )
        self.include_kfs = []
        self.__param_kw = ("*PARAMETER", "*PARAMETER_EXPRESSION")
        self.__acc_kwpre = TopoClsMap["nodes"] + TopoClsMap["elems"]
        self.__ori_kw_order = []
        self.__diff_kf = {"add": [], "del": [], "mod": []}
        self.diff_kf = MappingProxyType(self.__diff_kf)

    def __set_fieldconfig(self, FORMAT_TYPE="NORMAL"):
        self.__EntityCls_CardFields = copy.deepcopy(EntityCls_CardFields)
        self.__EntityCls_PagmFields = copy.deepcopy(EntityCls_PagmFields)
        if FORMAT_TYPE.upper() == "NORMAL":
            pass
        else:
            for fileds in [self.__EntityCls_CardFields]:
                for kw, value in fileds.items():
                    for each in value:
                        if FORMAT_TYPE.upper() == "I10":
                            _fieldlong = 10
                        if FORMAT_TYPE.upper() == "LONG":
                            _fieldlong = 20
                        each[:] = [_fieldlong if x < _fieldlong else x for x in each]
        for kw, value in self.__EntityCls_PagmFields.items():
            for _, field_setting in value.items():
                index = field_setting["index"]
                formater = field_setting["format"]
                if "e" in formater:
                    field_setting["format"] = formater.format(
                        self.__EntityCls_CardFields[kw][index[0]][index[1]] - 7
                    )
                if "d" in formater:
                    field_setting["format"] = formater.format(
                        self.__EntityCls_CardFields[kw][index[0]][index[1]]
                    )
                if "s" in formater:
                    field_setting["format"] = formater.format(
                        self.__EntityCls_CardFields[kw][index[0]][index[1]]
                    )
                if "f" in formater:
                    ...
        self.entityclass_cardfields = MappingProxyType(self.__EntityCls_CardFields)
        self.topoclass_cardfields = MappingProxyType(self.__EntityCls_CardFields)
        self.entityclass_pagmfields = MappingProxyType(self.__EntityCls_PagmFields)

    def __repr__(self) -> str:
        lines = []
        if self.__parsing_topo:
            lines.append("LsDynaEntity with:")
            for each in sum(self.__topocls_name__.values(), []):
                if each in self.keywords.keys():
                    lines.append(f"    {each}:".ljust(70) + f"{len(self.keywords[each])}")
        lines.append(f"  Entities:".ljust(70) + f"{len(self.keywords)}")
        lines.extend(
            [f"    {key}:".ljust(70) + f"{len(value)}" for key, value in self.keywords.items()]
        )
        return "\n".join(lines)

    @staticmethod
    def format_numeric2str(value: int | float, len_fomrat: int = 8):
        return format_numeric2str(value, len_fomrat)

    def filter_TopoDF_by_ids(
        self, et_type: str, ids: list[int], field: str = "id", return_asdf: bool = 0
    ):
        data = []
        if hasattr(self, et_type):
            _dd = getattr(self, et_type)
            for _k, _df in _dd.items():
                if not self.acc_filterbycache:
                    self.__filtercache = {}
                if _k not in self.__filtercache.keys():
                    self.__filtercache[_k] = _df.to_dict(orient="index")
                _k_f = "_".join([_k, field])
                if _k_f not in self.__filtercache.keys():
                    ix_map = defaultdict(list)
                    is_iterable = isinstance(_df[field].iloc[0], (list, tuple, np.ndarray))
                    if is_iterable:
                        for _ix, _r_ids in _df[field].to_dict().items():
                            for _r_id in _r_ids:
                                ix_map[_r_id].append(_ix)
                    else:
                        for _ix, _r_id in _df[field].to_dict().items():
                            ix_map[_r_id].append(_ix)
                    self.__filtercache[_k_f] = {"_ex_ia": ix_map, "_ex_ix": ix_map.keys()}
                _ex_ia = self.__filtercache[_k_f]["_ex_ia"]
                _ex_ix = self.__filtercache[_k_f]["_ex_ix"]
                _ex_dd = self.__filtercache[_k]
                _index = sum([_ex_ia[i] for i in ids if i in _ex_ix], [])
                pick = [_ex_dd[i] for i in _index]
                if pick:
                    if return_asdf:
                        pick = pd.DataFrame(pick)
                    data.append(pick)
        return data

    def __read_kwstr__(self, kf_lines: list[str], only_pre=0):
        _s_p = kf_lines[0].strip().split()
        kw_title: str = _s_p[0]
        kw_settings: str = " ".join(_s_p[1:])
        kw_lines: list[str] = kf_lines[1:]
        if (
            kw_title == "*ELEMENT_SOLID"
            and len(split_bywidth(kw_lines[0], [8] * 10)) > 2
            and len(self.__EntityCls_CardFields["*ELEMENT_SOLID"]) > 1
        ):
            self.__EntityCls_CardFields["*ELEMENT_SOLID"] = [[8] * 10]
            self.__EntityCls_PagmFields["*ELEMENT_SOLID"] = {
                "ID": {"index": [":", 0], "format": "", "info": "ID"},
                "PID": {"index": [":", 1], "format": "", "info": "PART ID"},
                "N1": {"index": [":", 2], "format": "", "info": "节点1"},
                "N2": {"index": [":", 3], "format": "", "info": "节点2"},
                "N3": {"index": [":", 4], "format": "", "info": "节点3"},
                "N4": {"index": [":", 5], "format": "", "info": "节点4"},
                "N5": {"index": [":", 6], "format": "", "info": "节点5"},
                "N6": {"index": [":", 7], "format": "", "info": "节点6"},
                "N7": {"index": [":", 8], "format": "", "info": "节点7"},
                "N8": {"index": [":", 9], "format": "", "info": "节点8"},
            }
        if kw_title.endswith("_TITLE") and kw_title not in self.__EntityCls_CardFields.keys():
            _t_ref = kw_title.replace("_TITLE", "")
            _ec = self.__EntityCls_CardFields
            _ec_k = _ec.keys()
            if _t_ref in _ec_k:
                _ec[kw_title] = [[80]] + _ec[_t_ref]
            else:
                _keys = sorted([x for x in _ec_k if _t_ref.startswith(x)], key=len, reverse=True)
                _ec[kw_title] = _ec[_keys[0]] if _keys else [[80]]
            _ep = self.__EntityCls_PagmFields
            _ep_k = _ep.keys()
            if _t_ref in _ep_k:
                _ep[kw_title] = copy.deepcopy(_ep[_t_ref])
            else:
                _keys = sorted([x for x in _ep_k if _t_ref.startswith(x)], key=len, reverse=True)
                _ep[kw_title] = _ep[_keys[0]] if _keys else {}
            _ep[kw_title].update({"NAME": {"index": [0, 0], "format": "", "info": "TITLE"}})
            for _, v in _ep[kw_title].items():
                _p_i = v["index"]
                if isinstance(_p_i[0], int):
                    _p_i[0] += 1
                elif isinstance(_p_i[0], str):
                    _s_s = _p_i[0].split(":")
                    _l_s = len(_s_s)
                    _f_a_0 = lambda x: 1 if x == "" else int(x) + 1
                    _f_a_1 = lambda x: "" if x == "" else int(x) + 1
                    if _l_s == 2:
                        _p_i[0] = f"{_f_a_0(_s_s[0])}:{_f_a_1(_s_s[1])}"
                    elif _l_s == 3:
                        _p_i[0] = f"{_f_a_0(_s_s[0])}:{_f_a_1(_s_s[1])}:{_s_s[2]}"
                    else:
                        raise ValueError("关键字字段索引设置错误")
        if kw_title == "*KEYWORD" and any([x in kw_settings for x in ["=Y", "=S"]]):
            if "I10" in kw_settings:
                self.__set_fieldconfig(FORMAT_TYPE="I10")
            if "LONG" in kw_settings:
                self.__set_fieldconfig(FORMAT_TYPE="LONG")
        if any([x in kw_settings for x in ["%", "+"]]):
            for fileds in [self.__EntityCls_CardFields]:
                if kw_title in fileds.keys():
                    for each in fileds[kw_title].values():
                        if "+" in kw_settings:
                            _fieldlong = 20
                        if "%" in kw_settings:
                            _fieldlong = 10
                        each = [_fieldlong if x < _fieldlong else x for x in each]
        if only_pre:
            return {
                "keyword": kw_title,
                "cards": kw_lines,
                "keyword_settings": kw_settings,
            }
        else:
            return LsDyna_ENTITY(
                self, keyword=kw_title, cards=kw_lines, keyword_settings=kw_settings
            )

    def __read_kwpreacc(self, kf_lines, kw_ranges):
        _c_kw = self.__acc_kwpre
        _ix = 1
        _c_kw_c = {k: [] for k in _c_kw}
        _c_kw_l = {k: -1 for k in _c_kw}
        for _s, _e in kw_ranges:
            _kw_l = kf_lines[_s]
            for k in _c_kw:
                if k == _kw_l.strip().split(" ", 1)[0]:
                    if _c_kw_l[k] == -1:
                        _c_kw_l[k] = [_kw_l]
                        _ix = _s
                    _c_kw_c[k].extend(kf_lines[_s + 1 : _e])
                    kf_lines[_s:_e] = ["pass"] * (_e - _s)
                    break
        kf_lines = (
            kf_lines[:_ix]
            + sum([_c_kw_l[k] + _c_kw_c[k] for k in _c_kw if _c_kw_c[k]], [])
            + kf_lines[_ix:]
        )
        kf_lines = [line for line in kf_lines if line != "pass"]
        star_lines = [index for index, line in enumerate(kf_lines) if line[0] == "*"]
        star_lines = star_lines + [star_lines[-1] + 1]
        kw_ranges = [(star_lines[i], star_lines[i + 1]) for i in range(len(star_lines) - 1)]
        return kf_lines, kw_ranges

    def read_kf(self, kfilepath, kwinkf=0, engine="bl", preacc=1):
        if engine == "bl":
            with open(kfilepath, encoding=self.encoding) as file:
                kf_lines = [line.upper() for line in file if line[0] != "$"]
            if not kf_lines[0].startswith("*KEYWORD"):
                raise ValueError("Missing *KEYWORD keyword")
            star_lines = [index for index, line in enumerate(kf_lines) if line[0] == "*"]
            star_lines = star_lines + [star_lines[-1] + 1]
            kw_ranges = [(star_lines[i], star_lines[i + 1]) for i in range(len(star_lines) - 1)]
            if preacc:
                kf_lines, kw_ranges = self.__read_kwpreacc(kf_lines, kw_ranges)
            kwinkf = kwinkf if kwinkf else {}
            if kwinkf:
                kwinkf = kwinkf
                _items = kw_ranges
            else:
                kwinkf = {}
                _items = tqdm(
                    kw_ranges,
                    desc="KW_ITEM ".ljust(30),
                    leave=True,
                    unit="",
                    bar_format="{l_bar}{bar:10}|     {n_fmt:>15}/{total_fmt:<16}",
                )
            for _s, _e in _items:
                entity = self.__read_kwstr__(kf_lines=kf_lines[_s:_e])
                _e_kw = entity.keyword
                if _e_kw not in kwinkf.keys():
                    kwinkf[_e_kw] = [entity]
                else:
                    kwinkf[_e_kw].append(entity)
                if _e_kw not in ["*KEYWORD", "*END"]:
                    self.__ori_kw_order.append(entity)
                if _e_kw == "*END":
                    break
                if _e_kw in self.__include_kw:
                    _include_kfs = []
                    if "PATH" in _e_kw:
                        for path in entity.cards:
                            path = pathlib.Path(path.replace("\n", ""))
                            if path.is_absolute():
                                ...
                            else:
                                path = pathlib.Path(kfilepath).parent / path
                            if path.is_dir():
                                _include_kfs.extend(list(path.glob("*.k")))
                            else:
                                _include_kfs.append(path)
                    elif "TRANSFORM" in _e_kw:
                        path = pathlib.Path(entity.cards[0].replace("\n", ""))
                        if path.is_absolute():
                            ...
                        else:
                            path = pathlib.Path(kfilepath).parent / path
                        _include_kfs = [path]
                    else:
                        for path in entity.cards:
                            path = pathlib.Path(path.replace("\n", ""))
                            if path.is_absolute():
                                ...
                            else:
                                path = pathlib.Path(kfilepath).parent / path
                            _include_kfs.append(path)
                    self.include_kfs.extend(_include_kfs)
                    for include_kf in _include_kfs:
                        self.read_kf(include_kf, kwinkf)
            kwinkf["*KEYWORD"] = [kwinkf["*KEYWORD"][0]]
            if not isinstance(_items, list):
                self.__ori_kw_order = kwinkf["*KEYWORD"] + self.__ori_kw_order
            _end = kwinkf.pop("*END", "")
            if _end:
                kwinkf["*END"] = [_end[-1]]
                if not isinstance(_items, list):
                    self.__ori_kw_order = self.__ori_kw_order + [_end[-1]]
            self.keywords: dict[str, LsDyna_ENTITY] = kwinkf
            return self

    def __acc_initcls(self, kw_settings, data_dicts, func, num=1, bar_title=""):
        batches = split_sequence(data_dicts, num)
        with ThreadPoolExecutor(max_workers=num) as executor:
            futures = [executor.submit(func, batch, kw_settings) for batch in batches]
            _obj = []
            with tqdm(
                total=len(futures),
                desc=f"    {bar_title.upper()} ".ljust(30),
                leave=True,
                unit="",
                bar_format="{l_bar}{bar:10}|     {n_fmt:>15}/{total_fmt:<16}{postfix:<16}",
            ) as pbar:
                for i, future in enumerate(as_completed(futures)):
                    _obj.extend(future.result())
                    pbar.update(1)
                    pbar.set_postfix_str(f"{len(_obj)}")
        return _obj

    def __create_nodes_batch(self, batch_data, kw_settings, progress_bar=0, bar_title=""):
        if progress_bar:
            batch_data = tqdm(
                batch_data,
                desc=f"    {bar_title} ".ljust(30),
                leave=True,
                unit="",
                bar_format="{l_bar}{bar:10}|     {n_fmt:>15}/{total_fmt:<16}",
            )
        return [LsDyna_NODE(self, **row, keyword_settings=kw_settings) for row in batch_data]

    def get_nodes(self, node_cardlines: list | str = "", kw_type="*NODE", is_init=1):
        self.__parsing_topo = 1
        self.__topocls_name__.update({"nodes": TopoClsMap["nodes"]})
        _all_cards = defaultdict(list)
        _d_c = ["id", "x", "y", "z"]
        kw_settings = ""
        if not node_cardlines:
            for _n_type in self.__topocls_name__["nodes"]:
                if _n_type in self.keywords:
                    node_cards = []
                    for _n in self.keywords[_n_type]:
                        node_cards.append(_n.cards)
                        if len(_n.keyword_settings) > len(kw_settings):
                            kw_settings = _n.keyword_settings
                    _all_cards[_n_type] = node_cards
            if not _all_cards:
                self.nodes = pd.DataFrame(columns=_d_c + ["obj"])
                return
        else:
            if isinstance(node_cardlines, str):
                node_cards = [node_cardlines.split("\n")]
            if isinstance(node_cardlines, list):
                node_cards = [node_cardlines]
            _all_cards[kw_type] = node_cards
            kw_settings = ""
        nodes_group_by_type = {}
        for _n_type, v in _all_cards.items():
            if _n_type in ["*NODE"]:
                _cf_0 = self.__EntityCls_CardFields["*NODE"][0]
                _f_b = lambda x: tqdm(
                    x,
                    desc=f"    {_n_type} ".ljust(30),
                    leave=False,
                    unit="",
                    bar_format="{l_bar}{bar:10}|     {n_fmt:>15}/{total_fmt:<16}",
                )
                _f_f = lambda x: x if x.strip() else ""
                _nodes = [
                    list(map(_f_f, split_bywidth(_l, _cf_0))) for _c in v for _l in _f_b(_c) if _l
                ]
                nodes = pd.DataFrame([n[:4] for n in _nodes], columns=_d_c)
                nodes["card1_add_fields"] = [
                    {key: vars for key, vars in zip(["TC", "RC"], each)}
                    for each in [x[4:] for x in _nodes]
                ]
                nodes["keyword"] = _n_type
                nodes["card_EX"] = ""
                nodes = nodes.astype(
                    dtype={"id": "int32", "x": "float64", "y": "float64", "z": "float64"},
                )
                nodes_group_by_type[_n_type] = nodes
            ...
        if not node_cardlines:
            if is_init:
                for _n_type, v in nodes_group_by_type.items():
                    if _n_type in ["*NODE"]:
                        data_dicts = v.to_dict(orient="records")
                        if self.__acc_initbythread:
                            _obj = self.__acc_initcls(
                                kw_settings,
                                data_dicts,
                                func=self.__create_nodes_batch,
                                bar_title=_n_type,
                            )
                        else:
                            _obj = self.__create_nodes_batch(
                                data_dicts, kw_settings, progress_bar=1, bar_title=_n_type
                            )
                        v["obj"] = _obj
                        self.keywords[_n_type] = v
            self.nodes: dict[str, pd.DataFrame] = nodes_group_by_type
        else:
            return nodes_group_by_type[kw_type]

    def __create_elems_batch(self, cls, batch_data, kw_settings, progress_bar=0, bar_title=""):
        if progress_bar:
            batch_data = tqdm(
                batch_data,
                desc=f"    {bar_title} ".ljust(30),
                leave=True,
                unit="",
                bar_format="{l_bar}{bar:10}|     {n_fmt:>15}/{total_fmt:<16}",
            )
        return [cls(self, **row, keyword_settings=kw_settings) for row in batch_data]

    def get_elems(self, elem_cardlines: list | str = "", kw_type="*ELEMENT_SOLID", is_init=1):
        self.__parsing_topo = 1
        self.__topocls_name__.update({"elems": TopoClsMap["elems"]})
        _all_cards = defaultdict(list)
        _d_c = ["id", "id_part", "id_nodes"]
        kw_settings = ""
        if not elem_cardlines:
            for _e_type in self.__topocls_name__["elems"]:
                if _e_type in self.keywords:
                    elem_cards = []
                    for _f_r in self.keywords[_e_type]:
                        elem_cards.append(_f_r.cards)
                        if len(_f_r.keyword_settings) > len(kw_settings):
                            kw_settings = _f_r.keyword_settings
                    _all_cards[_e_type] = elem_cards
            if not _all_cards:
                self.elems = pd.DataFrame(columns=_d_c)
                return
        else:
            if isinstance(elem_cardlines, str):
                elem_cards = [elem_cardlines.split("\n")]
            if isinstance(elem_cardlines, list):
                elem_cards = [elem_cardlines]
            _all_cards[kw_type] = elem_cards
            kw_settings = ""
        elems_group_by_type = {}
        for _e_type, _e_c in _all_cards.items():
            if _e_type in [
                "*ELEMENT_SOLID",
                "*ELEMENT_SHELL",
                "*ELEMENT_SHELL_THICKNESS",
                "*ELEMENT_BEAM",
                "*ELEMENT_BEAM_OFFSET",
                "*ELEMENT_BEAM_ORIENTATION",
            ]:
                _f_b = lambda x: tqdm(
                    x,
                    desc=f"    {_e_type} ".ljust(30),
                    leave=False,
                    unit="",
                    bar_format="{l_bar}{bar:10}|     {n_fmt:>15}/{total_fmt:<16}",
                )
                _f_f = lambda x: int(x) if x.strip() else ""
                _f_r = lambda _cf_0: [
                    list(map(_f_f, split_bywidth(_l, _cf_0)))
                    for _c in _e_c
                    for _l in _f_b(_c)
                    if _l
                ]
                _f_u = lambda x: list(dict.fromkeys([_id for _id in x if _id]))
                if _e_type in ["*ELEMENT_SOLID"]:
                    _cf_0 = self.__EntityCls_CardFields[_e_type][1]
                    _elems = _f_r(_cf_0)
                    try:
                        elems = pd.DataFrame(_elems[::2], columns=["id", "id_part"]).join(
                            pd.DataFrame({"id_nodes": [_f_u(i) for i in _elems[1::2]]})
                        )
                    except:
                        elems = pd.DataFrame(
                            [[x[0], x[1], _f_u(x[2:])] for x in _elems], columns=_d_c
                        )
                    elems["keyword"] = _e_type
                    elems["card_EX"] = ""
                if _e_type in ["*ELEMENT_SHELL", "*ELEMENT_SHELL_THICKNESS"]:
                    _cf_0 = self.__EntityCls_CardFields[_e_type][0]
                    if _e_type in ["*ELEMENT_SHELL_THICKNESS"]:
                        _f_r = lambda _cf_0: [
                            list(map(_f_f, split_bywidth(_l, _cf_0)))
                            for _c in _e_c
                            for _l in _f_b(_c[::2])
                            if _l
                        ]
                    _elems = _f_r(_cf_0)
                    elems = pd.DataFrame([[x[0], x[1], _f_u(x[2:])] for x in _elems], columns=_d_c)
                    elems["keyword"] = _e_type
                    elems["card_EX"] = ""
                    if _e_type in ["*ELEMENT_SHELL_THICKNESS"]:
                        elems["card_EX"] = [_l for _c in _e_c for _l in _c[1::2]]
                if _e_type in [
                    "*ELEMENT_BEAM",
                    "*ELEMENT_BEAM_OFFSET",
                    "*ELEMENT_BEAM_ORIENTATION",
                ]:
                    _cf_0 = self.__EntityCls_CardFields[_e_type][0]
                    if _e_type in ["*ELEMENT_BEAM_OFFSET", "*ELEMENT_BEAM_ORIENTATION"]:
                        _f_r = lambda _cf_0: [
                            list(map(_f_f, split_bywidth(_l, _cf_0)))
                            for _c in _e_c
                            for _l in _f_b(_c[::2])
                            if _l
                        ]
                    _elems = _f_r(_cf_0)
                    elems = pd.DataFrame([[x[0], x[1], x[2:4]] for x in _elems], columns=_d_c)
                    elems["card1_add_fields"] = [
                        {
                            key: vars
                            for key, vars in zip(["N3", "RT1", "RR1", "RT2", "RR2", "LOCAL"], each)
                        }
                        for each in [x[4:] for x in _elems]
                    ]
                    elems["keyword"] = _e_type
                    elems["card_EX"] = ""
                    if _e_type in ["*ELEMENT_BEAM_OFFSET", "*ELEMENT_BEAM_ORIENTATION"]:
                        elems["card_EX"] = [_l for _c in _e_c for _l in _c[1::2]]
                elems = elems.astype(dtype={"id": "int32", "id_part": "int32"})
                elems["id_nodes"] = elems["id_nodes"].apply(lambda x: [int(i) for i in x])
                elems_group_by_type[_e_type] = elems
        if not elem_cardlines:
            if is_init:
                for _e_type, _e_c in elems_group_by_type.items():
                    if _e_type in [
                        "*ELEMENT_SOLID",
                        "*ELEMENT_SHELL",
                        "*ELEMENT_SHELL_THICKNESS",
                        "*ELEMENT_BEAM",
                        "*ELEMENT_BEAM_OFFSET",
                        "*ELEMENT_BEAM_ORIENTATION",
                    ]:
                        if _e_type in ["*ELEMENT_SOLID"]:
                            _f_i = lambda b, k, p=0: self.__create_elems_batch(
                                cls=LsDyna_ELEMENT_SOLID,
                                batch_data=b,
                                kw_settings=k,
                                progress_bar=p,
                                bar_title=_e_type,
                            )
                        if _e_type in ["*ELEMENT_SHELL", "*ELEMENT_SHELL_THICKNESS"]:
                            _f_i = lambda b, k, p=0: self.__create_elems_batch(
                                cls=LsDyna_ELEMENT_SHELL,
                                batch_data=b,
                                kw_settings=k,
                                progress_bar=p,
                                bar_title=_e_type,
                            )
                        if _e_type in [
                            "*ELEMENT_BEAM",
                            "*ELEMENT_BEAM_OFFSET",
                            "*ELEMENT_BEAM_ORIENTATION",
                        ]:
                            _f_i = lambda b, k, p=0: self.__create_elems_batch(
                                cls=LsDyna_ELEMENT_BEAM,
                                batch_data=b,
                                kw_settings=k,
                                progress_bar=p,
                                bar_title=_e_type,
                            )
                        data_dicts = _e_c.to_dict(orient="records")
                        if self.__acc_initbythread:
                            _obj = self.__acc_initcls(
                                kw_settings, data_dicts, _f_i, bar_title=_e_type
                            )
                        else:
                            _obj = _f_i(data_dicts, kw_settings, 1)
                        _e_c["obj"] = _obj
                        self.keywords[_e_type] = _e_c
            self.elems: dict[str, pd.DataFrame] = elems_group_by_type
        else:
            return elems_group_by_type[kw_type]

    def get_parts(self, part_cardlines: list | str = "", kw_type="*PART", is_init=1):
        self.__parsing_topo = 1
        self.__topocls_name__.update({"parts": TopoClsMap["parts"]})
        _all_cards = defaultdict(list)
        _d_c = ["id", "name", "id_sec", "id_mat", "card2_add_fields"]
        kw_settings = ""
        if not part_cardlines:
            for _p_type in self.__topocls_name__["parts"]:
                if _p_type in self.keywords:
                    part_cards = []
                    for _p in self.keywords[_p_type]:
                        part_cards.append(_p.cards)
                        if len(_p.keyword_settings) > len(kw_settings):
                            kw_settings = _p.keyword_settings
                    _all_cards[_p_type] = part_cards
            if not _all_cards:
                self.parts = pd.DataFrame(columns=_d_c + ["obj"]).astype(
                    dtype={"id": "int32", "id_sec": "int32", "id_mat": "int32"}
                )
                return
        else:
            if isinstance(part_cardlines, str):
                part_cards = [part_cardlines.split("\n")]
            if isinstance(part_cardlines, list):
                part_cards = [part_cardlines]
            _all_cards[kw_type] = part_cards
            kw_settings = ""
        parts_group_by_type = {}
        for _p_type, _p_c in _all_cards.items():
            if _p_type in ["*PART"]:
                _parts = []
                _cf_1 = self.__EntityCls_CardFields["*PART"][1]
                _f_f = lambda x: int(x) if x.strip() else ""
                _parts = [
                    _l.strip() if index % 2 == 0 else list(map(_f_f, split_bywidth(_l, _cf_1)))
                    for index, _l in enumerate([_l for _c in _p_c for _l in _c])
                ]
                parts = pd.DataFrame(_parts[::2], columns=["name"])
                parts[["id", "id_sec", "id_mat"]] = pd.DataFrame([x[:3] for x in _parts[1::2]])
                parts["card2_add_fields"] = [
                    {
                        key: vars
                        for key, vars in zip(
                            ["EOSID", "HGID", "GRAV", "ADPOPT", "TMID"],
                            each,
                        )
                    }
                    for each in [x[3:] for x in _parts[1::2]]
                ]
                parts["keyword"] = _p_type
                parts["card_EX"] = ""
                parts = parts.astype(dtype={"id": "int32"})
                for func, keys in zip([int], [["id_sec", "id_mat"]]):
                    for key in keys:
                        parts[key] = parts[key].apply(lambda x: func(x) if not x == "" else x)
                parts_group_by_type[_p_type] = parts
            ...
        if not part_cardlines:
            if is_init:
                for _p_type, _p_c in parts_group_by_type.items():
                    if _p_type in ["*PART"]:
                        data_dicts = _p_c.to_dict(orient="records")
                        _p_c["obj"] = [
                            LsDyna_PART(self, **row, keyword_settings=kw_settings)
                            for row in tqdm(
                                data_dicts,
                                desc=f"    {_p_type} ".ljust(30),
                                leave=True,
                                unit="",
                                bar_format="{l_bar}{bar:10}|     {n_fmt:>15}/{total_fmt:<16}",
                            )
                        ]
                        self.keywords[_p_type] = _p_c
            self.parts: pd.DataFrame = parts_group_by_type
        else:
            return parts_group_by_type[kw_type]

    def get_define_curve(
        self, curve_cardlines: list | str = "", kw_type="*DEFINE_CURVE", is_init=1, replace_param=1
    ):
        self.__parsing_topo = 1
        self.__topocls_name__.update({"define_curve": TopoClsMap["define_curve"]})
        _all_cards = defaultdict(list)
        _d_c = ["id", "sidr", "sfa", "sfo", "offa", "offo", "dattyp", "lcint", "x", "y"]
        kw_settings = ""
        if not curve_cardlines:
            for _c_type in self.__topocls_name__["define_curve"]:
                if _c_type in self.keywords:
                    curve_cards = []
                    for _c in self.keywords[_c_type]:
                        curve_cards.append(_c.cards)
                        if len(_c.keyword_settings) > len(kw_settings):
                            kw_settings = _c.keyword_settings
                    _all_cards[_c_type] = curve_cards
            if not _all_cards:
                self.curves = pd.DataFrame(columns=_d_c + ["obj"])
                return
        else:
            if isinstance(curve_cardlines, str):
                curve_cards = [curve_cardlines.split("\n")]
            if isinstance(curve_cardlines, list):
                if len(curve_cardlines) == 2 and isinstance(curve_cardlines[1], list):
                    curve_cardlines = [curve_cardlines[0], *curve_cardlines[1]]
                curve_cards = [curve_cardlines]
            _all_cards[kw_type] = curve_cards
        curves_group_by_type = {}
        for _c_type, v in _all_cards.items():
            _param_mapping = (
                dict(zip(self.parameters["names"], self.parameters["vals_s"]))
                if replace_param
                else {}
            )
            if _c_type in ["*DEFINE_CURVE"]:
                curves = []
                _cf_0 = self.__EntityCls_CardFields["*DEFINE_CURVE"][0]
                _cf_1 = self.__EntityCls_CardFields["*DEFINE_CURVE"][1]
                for _c_c in v:
                    card0 = [
                        func(var)
                        for var, func in zip(
                            split_bywidth(_c_c[0], _cf_0),
                            [lambda x: int(x) if x.strip() else ""] * 2
                            + [lambda x: float(x) if x.strip() else ""] * 4
                            + [lambda x: int(x) if x.strip() else ""] * 2,
                        )
                    ]
                    if len(card0) < 8:
                        card0 += [""] * (8 - len(card0))
                    card1 = [
                        list(map(float, split_bywidth(_l, _cf_1, _param_mapping)))
                        for _l in _c_c[1:]
                        if _l
                    ]
                    curves.append(
                        [
                            *card0,
                            [x[0] for x in card1],
                            [np.nan if len(x) < 2 else x[1] for x in card1],
                        ]
                    )
                curves = pd.DataFrame(curves, columns=_d_c)
                curves["keyword"] = _c_type
                curves["card_EX"] = ""
                curves = curves.astype(dtype={"id": "int32"})
                for func, keys in zip([int, float], [["lcint"], ["sfa", "sfo", "offa", "offo"]]):
                    for key in keys:
                        curves[key] = curves[key].apply(lambda x: func(x) if not x == "" else x)
                curves_group_by_type[_c_type] = curves
            ...
        if not curve_cardlines:
            if is_init:
                for _c_type, v in curves_group_by_type.items():
                    if _c_type in ["*DEFINE_CURVE"]:
                        data_dicts = v.to_dict(orient="records")
                        v["obj"] = [
                            LsDyna_DEFINE_CURVE(self, **row, keyword_settings=kw_settings)
                            for row in tqdm(
                                data_dicts,
                                desc=f"    {_c_type} ".ljust(30),
                                leave=True,
                                unit="",
                                bar_format="{l_bar}{bar:10}|     {n_fmt:>15}/{total_fmt:<16}",
                            )
                        ]
                        self.keywords[_c_type] = v
            self.curves: pd.DataFrame = curves_group_by_type
        else:
            return curves_group_by_type[kw_type]

    def get_set_list(self, set_cardlines: list | str = "", kw_type="*SET_NODE_LIST", is_init=1):
        self.__parsing_topo = 1
        self.__topocls_name__.update({"set_list": TopoClsMap["set_list"]})
        _all_cards = defaultdict(list)
        _d_c = ["id", "da1", "da2", "da3", "da4", "solver", "nids"]
        kw_settings = ""
        if not set_cardlines:
            for _s_type in self.__topocls_name__["set_list"]:
                if _s_type in self.keywords:
                    set_cards = []
                    for _s in self.keywords[_s_type]:
                        set_cards.append(_s.cards)
                        if len(_s.keyword_settings) > len(kw_settings):
                            kw_settings = _s.keyword_settings
                    _all_cards[_s_type] = set_cards
            if not set_cards:
                self.sets = pd.DataFrame(columns=_d_c + ["obj"])
                return
        else:
            if isinstance(set_cardlines, str):
                set_cards = [set_cardlines.split("\n")]
            if isinstance(set_cardlines, list):
                if len(set_cardlines) == 2 and isinstance(set_cardlines[1], list):
                    set_cardlines = [set_cardlines[0], *set_cardlines[1]]
                set_cards = [set_cardlines]
            _all_cards[kw_type] = set_cards
        sets_group_by_type = {}
        for _s_type, v in _all_cards.items():
            if _s_type in [
                "*SET_NODE",
                "*SET_PART",
                "*SET_SHELL",
                "*SET_NODE_LIST",
                "*SET_PART_LIST",
                "*SET_SHELL_LIST",
                "*SET_NODE_TITLE",
                "*SET_PART_TITLE",
                "*SET_SHELL_TITLE",
                "*SET_NODE_LIST_TITLE",
                "*SET_PART_LIST_TITLE",
                "*SET_SHELL_LIST_TITLE",
            ]:
                _setslist = []
                _cf_0 = self.__EntityCls_CardFields[_s_type.replace("_TITLE", "")][0]
                _cf_1 = self.__EntityCls_CardFields[_s_type.replace("_TITLE", "")][1]
                for _c in v:
                    if _s_type.endswith("_TITLE"):
                        _c = _c[1:]
                    card0 = [
                        func(var)
                        for var, func in zip(
                            split_bywidth(_c[0], _cf_0),
                            [lambda x: int(x) if x.strip() else ""]
                            + [lambda x: float(x) if x.strip() else ""] * 4
                            + [lambda x: str(x) if x.strip() else ""],
                        )
                    ]
                    if len(card0) < 6:
                        card0 += [""] * (6 - len(card0))
                    card1 = [split_bywidth(_l, _cf_1) for _l in _c[1:]]
                    _setslist.append([*card0, card1])
                _setslist = pd.DataFrame(_setslist, columns=_d_c)
                _setslist["keyword"] = _s_type.replace("_TITLE", "")
                _setslist["card_EX"] = ""
                _setslist = _setslist.astype(dtype={"id": "int32"})
                _setslist["nids"] = _setslist["nids"].apply(
                    lambda x: [np.int32(i).tolist() for i in x]
                )
                sets_group_by_type[_s_type] = _setslist
            ...
        if not set_cardlines:
            if is_init:
                for _s_type, v in sets_group_by_type.items():
                    if _s_type in [
                        "*SET_NODE",
                        "*SET_NODE_LIST",
                        "*SET_PART",
                        "*SET_PART_LIST",
                        "*SET_SHELL",
                        "*SET_SHELL_LIST",
                        "*SET_NODE_TITLE",
                        "*SET_PART_TITLE",
                        "*SET_SHELL_TITLE",
                        "*SET_NODE_LIST_TITLE",
                        "*SET_PART_LIST_TITLE",
                        "*SET_SHELL_LIST_TITLE",
                    ]:
                        data_dicts = v.to_dict(orient="records")
                        v["obj"] = [
                            LsDyna_SET_LIST(self, **row, keyword_settings=kw_settings)
                            for row in tqdm(
                                data_dicts,
                                desc=f"    {_s_type} ".ljust(30),
                                leave=True,
                                unit="",
                                bar_format="{l_bar}{bar:10}|     {n_fmt:>15}/{total_fmt:<16}",
                            )
                        ]
                        self.keywords[_s_type] = v
            self.sets: dict[str, pd.DataFrame] = sets_group_by_type
        else:
            return sets_group_by_type[kw_type]

    def collect_portion_MAT(self):
        mats = []
        for each in self.keywords.keys():
            if each.startswith("*MAT"):
                for mat in self.keywords[each]:
                    mats.append([mat["MID"], mat.keyword])
        self.mats = pd.DataFrame(mats, columns=["id", "type"]).astype(dtype={"id": "int32"})

    def collect_portion_SECTION(self):
        sections = []
        for each in self.keywords.keys():
            if each.startswith("*SEC"):
                for sec in self.keywords[each]:
                    sections.append([sec["SECID"], sec.keyword])
        self.sections = pd.DataFrame(sections, columns=["id", "type"]).astype(dtype={"id": "int32"})

    def collect_PARAMETER(self):
        parameters = {"names": [], "vals_s": [], "keyword": [], "type": []}
        for k in self.__param_kw:
            if k in self.keywords.keys():
                for p in self.keywords[k]:
                    if len(p.cards) == 1:
                        parameters["names"].append(p["K"].strip().split()[-1])
                        parameters["type"].append(p["K"].strip().split()[0])
                        parameters["vals_s"].append(p["V"].strip())
                        parameters["keyword"].append(p.keyword)
                    else:
                        _f_ep = lambda x: [j.strip() for i in x for j in i[1]]
                        parameters["names"].extend([x.split()[-1] for x in _f_ep(p["K"])])
                        parameters["type"].extend([x.split()[0] for x in _f_ep(p["K"])])
                        parameters["vals_s"].extend(_f_ep(p["V"]))
                        parameters["keyword"].extend([p.keyword] * len(_f_ep(p["V"])))
        parameters = pd.DataFrame(parameters)

        def _f_f(x):
            try:
                return float(x)
            except:
                return x

        parameters["vals_n"] = parameters["vals_s"].apply(_f_f)
        self.parameters = parameters

    def remove_kw(self, kw: str, at_index: int):
        if self.keywords.get(kw, False) is not False and at_index < len(self.keywords[kw]):
            _kw_container = self.keywords[kw]
            if kw in sum(self.__topocls_name__.values(), []):
                _delkw = _kw_container.iloc[at_index]
                _kw_container.drop(at_index, axis=0, inplace=True)
                _kw_container.reset_index(drop=True, inplace=True)
            else:
                _delkw = _kw_container.pop(at_index)
                self.__ori_kw_order.remove(_delkw)
            if len(_kw_container) == 0:
                self.keywords.pop(kw)
            self.__diff_kf["del"].append({_delkw.keyword: _delkw})
            self.__filtercache = {}
            return {"del": _delkw}
        else:
            return "删除失败"

    def __update_kwdf__(self, newkwobj):
        _reverse = {v: k for k, vs in self.__topocls_name__.items() for v in vs}
        if newkwobj.keyword in _reverse.keys():
            _func = getattr(self, f"get_{_reverse[newkwobj.keyword].lower()}")
            _pd_newcols = _func(newkwobj.str_cardsonly, newkwobj.keyword)
        _pd_newkw = pd.DataFrame([newkwobj], columns=["obj"]).join(_pd_newcols)
        return _pd_newkw

    def insert_kw(
        self,
        newkwobj,
        at_index,
        method: str = "add",
    ):
        kw = newkwobj.keyword
        if self.keywords.get(kw, False) is not False and at_index < len(self.keywords[kw]):
            _kw_container = self.keywords[kw]
            if kw in sum(self.__topocls_name__.values(), []):
                if method == "add":
                    _pd_newkw = self.__update_kwdf__(newkwobj)
                    _kw_container.loc[len(_kw_container)] = [0] * len(_kw_container.loc[0])
                    _kw_container.iloc[:] = pd.concat(
                        [_kw_container.iloc[:at_index], _pd_newkw, _kw_container.iloc[at_index:-1]],
                        axis=0,
                        ignore_index=True,
                    )
                if method == "replace":
                    _kw_container.iloc[at_index] = newkwobj
                self.__filtercache = {}
            elif isinstance(_kw_container, list):
                _ix = self.__ori_kw_order.index(_kw_container[at_index])
                if method == "add":
                    _kw_container.insert(at_index, newkwobj)
                    self.__ori_kw_order.insert(_ix, newkwobj)
                if method == "replace":
                    _kw_container[at_index] = newkwobj
                    self.__ori_kw_order[_ix] = newkwobj
        else:
            __end = {"*END": []}
            if self.keywords.get("*END", False):
                __end = self.keywords.pop("*END")
            self.keywords.update({kw: [newkwobj]})
            self.keywords.update(__end)
            self.__ori_kw_order.insert(-1, newkwobj)
            method = "add"
        self.__diff_kf[method].append({kw: newkwobj})
        return {method: {"at_index": at_index, "keyword": kw, "obj": newkwobj}}

    def save_kf(self, path):
        with open(path, "w") as file:
            if self.__parsing_topo:
                passed_k = []
                for _kw_e in self.__ori_kw_order:
                    k = _kw_e.keyword
                    if k in sum(
                        [
                            self.__topocls_name__[x]
                            for x in ["nodes", "elems"]
                            if self.__topocls_name__.get(x, False)
                        ],
                        [],
                    ):
                        if isinstance(self.keywords[k], pd.DataFrame):
                            if k not in passed_k:
                                kwobj_container = self.keywords[k]["obj"]
                                file.write(
                                    k + " " + kwobj_container.iloc[0].keyword_settings + "\n"
                                )
                                for each in kwobj_container:
                                    file.write(each.str_cardsonly)
                            else:
                                continue
                        else:
                            file.write(_kw_e.str)
                    elif k in sum(
                        [
                            self.__topocls_name__[x]
                            for x in ["parts", "define_curve", "set_list"]
                            if self.__topocls_name__.get(x, False)
                        ],
                        [],
                    ):
                        if isinstance(self.keywords[k], pd.DataFrame):
                            if k not in passed_k:
                                for each in self.keywords[k]["obj"]:
                                    file.write(each.str)
                            else:
                                continue
                        else:
                            file.write(_kw_e.str)
                    elif k in self.__include_kw:
                        pass
                    else:
                        file.write(_kw_e.str)
                    passed_k.append(k)
            else:
                for _kw_e in self.__ori_kw_order:
                    k = _kw_e.keyword
                    if k in self.__include_kw:
                        pass
                    else:
                        file.write(_kw_e.str)
        return path


if __name__ == "__main__":
    import time, cProfile, pstats

    start = time.time()
    j = bl_keyfile(
        r"C:\Users\breez\Desktop\kf\k\0.k", parsing_topo=1, is_init=1, acc_initbythread=0
    )
    end = time.time()
    print(end - start)
