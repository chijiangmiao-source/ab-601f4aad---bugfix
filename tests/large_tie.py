"""题述大规模同优场景的规律化数据生成（171 击中、226 候选）。

结构：
* 击中 h0..h170，位置 0..170；
* a-split=(h0,h57)、z-outer=(h0,h170)，残差均为 0；
* 28 个嵌套六击中单元 g00..g27：单元 g 的六个局部点从左到右为
  h(1+2g)、h(2+2g)、h(166-4g)、h(167-4g)、h(168-4g)、h(169-4g)；
* 每单元 8 条候选 gXXe0..gXXe7，连接局部下标对
  (0,1)、(0,3)、(0,5)、(1,2)、(1,4)、(2,3)、(3,4)、(4,5)。

最优结果可独立复核：
* 各单元左点落在 1..56、右点落在 58..169，两两不交且单元间只嵌套不交叉；
  每单元恰有 4 个可用的非交叉完美匹配：
  {e0,e5,e7}、{e1,e3,e7}、{e2,e3,e6}、{e2,e4,e5}。
* z-outer=(h0,h170) 包裹全部单元，与任意单元选择相容 => 4^28 个方案
  （170 点击中、残差 0、h57 未配对）。
* a-split=(h0,h57) 与所有跨越 57/58 间隙的弧（e1..e4）交叉，只相容于
  每单元唯一的“左右各自相邻”匹配 {e0,e5,e7} => 恰 1 个方案
  （170 点击中、残差 0、h170 未配对）。
* 总计 4^28 + 1 个最优方案；两条外层弧分别只出现在 4^28 与 1 个方案中，
  均非必选；规范序列按 id 字典序从 a-split 开始。
"""

from __future__ import annotations

N_HITS = 171
N_UNITS = 28
EXPECTED_OPTIMAL_COUNT = 4 ** N_UNITS + 1

_PAIRS = [(0, 1), (0, 3), (0, 5), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]


def large_tie_payload() -> dict:
    """生成完整审计请求体。"""
    hits = [{"id": f"h{k}", "position": k} for k in range(N_HITS)]
    candidates = [
        {"id": "a-split", "left_endpoint": "h0", "right_endpoint": "h57", "residual": 0},
        {"id": "z-outer", "left_endpoint": "h0", "right_endpoint": "h170", "residual": 0},
    ]
    for g in range(N_UNITS):
        local = [1 + 2 * g, 2 + 2 * g] + [166 - 4 * g + t for t in range(4)]
        for e, (u, v) in enumerate(_PAIRS):
            candidates.append(
                {
                    "id": f"g{g:02d}e{e}",
                    "left_endpoint": f"h{local[u]}",
                    "right_endpoint": f"h{local[v]}",
                    "residual": 0,
                }
            )
    return {"hits": hits, "candidates": candidates}


def expected_candidate_ids() -> list[str]:
    ids = ["a-split", "z-outer"]
    for g in range(N_UNITS):
        ids += [f"g{g:02d}e{e}" for e in range(8)]
    return ids


def expected_canonical_sequence() -> list[str]:
    """规范方案：先 a-split，再 g00e0..g27e0，随后 g27..g00 每组 e5、e7。"""
    seq = ["a-split"] + [f"g{g:02d}e0" for g in range(N_UNITS)]
    for g in range(N_UNITS - 1, -1, -1):
        seq += [f"g{g:02d}e5", f"g{g:02d}e7"]
    return seq
