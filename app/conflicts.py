"""全部最小零概率观测组的精确求解。

给定同一 ROBDD 上的 m≤16 个读数（每个读数是一个 BDD 字面量：观测
id 读 true 取其 BDD 节点，读 false 取其取反），一组读数的合取在事件
独立先验下的发生概率即对应 BDD 节点的加权概率。冲突组 C 满足：

  P(∧_{i∈C} 读数 i) = 0，且删去组内任一项后概率严格大于 0。

等价地，按集合包含关系极小（minimal）的零概率读数子集（MUS）。

判定完全复用决策图，不枚举事件赋值。采用经典的 MARCO 式极小不可满足
子集枚举（seed 的 grow/shrink）：
  - 掩码 `blocked` 标记两类不必再考察的读数子集——已找到极小零组的全部
    超集（含零组，非极小）、已找到极大正概率子集的全部子集（正概率，
    非零组）；
  - 反复取一个未被标记的 seed：其合取概率为 0 就贪心“收缩”（逐项尝试
    删除，删除后仍为零概率则保留删除）得到极小零组，并标记其全部超集；
    概率为正就贪心“扩张”（逐项尝试并入，并入后仍为正概率则保留）得到
    极大正概率子集，并标记其全部子集；
  - 任一掩码至多成为一次 seed，每次扩张/收缩至多 m 次合取判定；合取
    节点按掩码记忆化，同掩码不重复构建。
合取是否零概率完全由 BDD 加权概率判定：结构矛盾的合取恒为终端 0，
p=0/1、1/1 先验造成的零概率体现为非终端节点概率为 0，两者都被识别。

结果组数超过 MAX_CONFLICTS（5000）时抛出 ConflictLimitExceeded；
合取构建复用同一 ROBDD 的 50000 非终端节点限额，超限由 ROBDD 抛出
NodeLimitExceeded。
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Sequence

from .bdd import ROBDD

# 允许返回的冲突组数量上限
MAX_CONFLICTS = 5000


class ConflictLimitExceeded(ValueError):
    """全部最小零概率观测组超过组数上限。"""


def minimal_zero_probability_groups(
    bdd: ROBDD,
    literals: Sequence[int],
    probs: List[Fraction],
) -> List[List[int]]:
    """求全部极小零概率读数子集，返回读数下标的列表的列表。

    literals[i] 为第 i 个读数（已含取值正负）的 BDD 节点；probs 为按
    BDD 变量序排列的事件先验。返回的每组只含读数下标，组内按下标升序，
    组间按 (长度, 下标序列) 排序。组数超过 MAX_CONFLICTS 抛
    ConflictLimitExceeded；构建合取时节点超限由 bdd 抛出
    NodeLimitExceeded。
    """
    m = len(literals)
    full_mask = (1 << m) - 1

    # 合取节点记忆化 + 增量概率表：新增 BDD 节点后只向后补算概率，
    # 避免每次判定都对全唯一表重新加权。
    conj_node: Dict[int, int] = {0: 1}
    node_prob = [Fraction(0), Fraction(1)]

    def conjunction(mask: int) -> int:
        cached = conj_node.get(mask)
        if cached is not None:
            return cached
        # 去掉最低置位：父状态小于 mask，递归必已记忆（seed/收缩均按
        # “更小掩码”访问），再合入该字面量
        low_bit = mask & -mask
        i = low_bit.bit_length() - 1
        prev = conjunction(mask ^ low_bit)
        node = 0 if prev == 0 else bdd.and_(prev, literals[i])
        conj_node[mask] = node
        return node

    def is_zero(mask: int) -> bool:
        node = conjunction(mask)
        bdd.extend_probabilities(probs, node_prob)
        return node_prob[node] == 0

    # 快速判定常见情形：全部读数可同时成立 ⟺ 全集合取概率为正，无冲突
    if not is_zero(full_mask):
        return []

    # MARCO 式枚举。blocked 标记两类“不必再考察”的掩码：
    #   a) 已找到极小零组的全部超集（含零组，非极小）；
    #   b) 已找到极大正概率子集的全部子集（正概率，非零组）。
    # 每轮取一个未标记 seed：概率为正就贪心扩张成极大正概率子集并标记其
    # 全部子集；概率为 0 就贪心收缩成极小零组并标记其全部超集。任一掩码
    # 至多成为一次 seed；每次扩张/收缩至多 m 次合取判定。
    blocked = bytearray(1 << m)
    zero_masks: List[int] = []

    def next_seed() -> int:
        """取最大的未标记掩码；全体掩码考察完毕时返回 0（空集恒真）。"""
        mask = full_mask
        while mask and blocked[mask]:
            mask -= 1
        return mask

    seed = next_seed()
    while seed:
        if is_zero(seed):
            # 贪心收缩：逐项尝试删除读数，删除后仍为零概率就永久删除，
            # 最终得到极小零组（顺序无关：收缩结束时删任一项概率都为正）。
            # 每轮只遍历当前 mus 中的位，删除后重新扫描，避免把已删位加回。
            mus = seed
            changed = True
            while changed:
                changed = False
                bit = mus
                while bit:
                    b = bit & -bit
                    bit ^= b
                    trial = mus ^ b
                    if is_zero(trial):
                        mus = trial
                        changed = True
                        break  # mus 已变，重新从最低位开始扫描

            zero_masks.append(mus)
            if len(zero_masks) > MAX_CONFLICTS:
                raise ConflictLimitExceeded(
                    f"全部最小零概率观测组超过 {MAX_CONFLICTS} 组的限制"
                )

            # 标记 mus 的全部超集（枚举 ~mus 在全集内的子掩码 rest）
            free = full_mask ^ mus
            rest = free
            while True:
                blocked[mus | rest] = 1
                if rest == 0:
                    break
                rest = (rest - 1) & free
        else:
            # 贪心扩张：逐项尝试并入 seed 之外的读数，并入后仍为正概率就
            # 保留，最终得到极大正概率子集 mss
            mss = seed
            bit = full_mask ^ mss
            while bit:
                b = bit & -bit
                bit ^= b
                trial = mss | b
                if not is_zero(trial):
                    mss = trial

            # 标记 mss 的全部子集（枚举 mss 的子掩码 sub）
            sub = mss
            while True:
                blocked[sub] = 1
                if sub == 0:
                    break
                sub = (sub - 1) & mss

        seed = next_seed()

    # 组内下标升序（位序即按下标）；组间按 (长度, 下标序列) 排序
    groups = [
        [i for i in range(m) if mask >> i & 1]
        for mask in zero_masks
    ]
    groups.sort(key=lambda idxs: (len(idxs), idxs))
    return groups
