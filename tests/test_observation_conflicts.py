"""POST /v1/observation-conflicts：全部最小零概率观测组的语义与 422 校验。

用穷举真值表（枚举全部事件状态）作为参考实现做交叉验证。
"""

import itertools
import random
from fractions import Fraction

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------- 穷举真值表参考实现 ----------

def brute_conflicts(ev_specs, gate_specs, top, observations):
    """枚举全部事件赋值，返回全部极小零概率观测组（观测 id 列表）。

    一组读数的合取概率为 0 且删去任一项后概率严格为正即为极小零组。
    """
    ev_ids = sorted(ev_specs)
    probs = {e: Fraction(ev_specs[e]) for e in ev_ids}
    gates = {g[0]: (g[1], g[2]) for g in gate_specs}

    def evaluate(state, node):
        if node in state:
            return state[node]
        op, inputs = gates[node]
        vals = [evaluate(state, i) for i in inputs]
        if op == "AND":
            return all(vals)
        if op == "OR":
            return any(vals)
        return not vals[0]

    obs_ids = sorted(observations)
    values = [observations[i] for i in obs_ids]

    def subset_probability(sub):
        if not sub:
            return Fraction(1)
        total = Fraction(0)
        for bits in itertools.product((0, 1), repeat=len(ev_ids)):
            state = dict(zip(ev_ids, (bool(b) for b in bits)))
            if all(
                evaluate(state, obs_ids[i]) == values[i] for i in sub
            ):
                weight = Fraction(1)
                for e in ev_ids:
                    weight *= probs[e] if state[e] else (1 - probs[e])
                total += weight
        return total

    groups = []
    for k in range(1, len(obs_ids) + 1):
        for combo in itertools.combinations(range(len(obs_ids)), k):
            sub = frozenset(combo)
            if subset_probability(sub) == 0 and all(
                subset_probability(sub - {j}) > 0 for j in sub
            ):
                groups.append([obs_ids[i] for i in combo])
    groups.sort(key=lambda ids: (len(ids), ids))
    return groups


def make_body(ev_specs, gate_specs, top, observations):
    return {
        "events": [{"id": k, "p": v} for k, v in ev_specs.items()],
        "gates": [
            {"id": i, "op": op, "inputs": ins}
            for i, op, ins in gate_specs
        ],
        "top": top,
        "observations": observations,
    }


MOTOR_EV = {
    "motor_overheat": "1/100",
    "coolant_loss": "2/100",
    "pump_fail": "1/200",
}
MOTOR_GATES = [
    ("cooling_fail", "OR", ["coolant_loss", "pump_fail"]),
    ("top", "OR", ["motor_overheat", "cooling_fail"]),
]


# ---------- 基本语义 ----------

def test_all_readings_compatible_returns_empty():
    # 全部读数可同时成立：先验为正 => conflicts 为空
    body = make_body(
        MOTOR_EV, MOTOR_GATES, "top",
        {"motor_overheat": False, "top": True},
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {"conflicts": []}


def test_structural_pair_conflict():
    # a=true 与 NOT a=true 同时成立在结构上不可能：唯一极小零组为两者
    body = make_body(
        {"a": "1/2"},
        [("na", "NOT", ["a"])], "na",
        {"a": True, "na": True},
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {"conflicts": [["a", "na"]]}


def test_zero_and_one_prior_single_conflicts():
    # p=0/1 的事件读 true、p=1/1 的事件读 false 各自单独即零概率
    body = make_body(
        {"a": "0/1", "b": "1/1"},
        [("g", "OR", ["a", "b"])], "g",
        {"a": True, "b": False},
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {"conflicts": [["a"], ["b"]]}


def test_p_zero_event_read_true_is_single_conflict():
    body = make_body({"a": "0/1"}, [], "a", {"a": True})
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200
    assert r.json() == {"conflicts": [["a"]]}


def test_p_one_event_read_false_is_single_conflict():
    body = make_body({"a": "1/1"}, [], "a", {"a": False})
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200
    assert r.json() == {"conflicts": [["a"]]}


def test_p_zero_read_false_is_fine():
    body = make_body({"a": "0/1"}, [], "a", {"a": False})
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200
    assert r.json() == {"conflicts": []}


def test_gate_reading_conflict():
    # g = a AND b 读 true，同时 a 读 false：门读数按共享状态求值 => 零概率
    body = make_body(
        {"a": "1/2", "b": "1/3"},
        [("g", "AND", ["a", "b"])], "g",
        {"g": True, "a": False, "b": True},
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200, r.text
    # [a=false, g=true] 零概率；b=true 无关（删去 a 或 g 概率都为正）
    assert r.json() == {"conflicts": [["a", "g"]]}


def test_not_gate_conflict_pair():
    # ng = NOT g（g=a AND b）：g=true 与 ng=true 不可能同时成立，
    # 额外的无关读数 c=true 不进冲突组
    body = make_body(
        {"a": "1/2", "b": "1/3", "c": "1/5"},
        [("g", "AND", ["a", "b"]), ("ng", "NOT", ["g"])], "ng",
        {"g": True, "ng": True, "c": True},
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {"conflicts": [["g", "ng"]]}


def test_genuine_three_way_conflict():
    # g = a XOR b = (a OR b) AND NOT(a AND b)。
    # g=true、a=true、b=true 三者合取为 0，但任意两者合取概率为正 =>
    # 极小零组恰为三项（NOT 门照常生效）
    body = make_body(
        {"a": "1/2", "b": "1/3"},
        [("o", "OR", ["a", "b"]),
         ("ad", "AND", ["a", "b"]),
         ("n", "NOT", ["ad"]),
         ("g", "AND", ["o", "n"])],
        "g",
        {"g": True, "a": True, "b": True},
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {"conflicts": [["a", "b", "g"]]}


def test_sorting_within_and_between_groups():
    # 4 个独立的零概率单点 + 一对结构矛盾：组间按长度、再按字典序
    body = make_body(
        {"a": "0/1", "b": "0/1", "c": "1/2", "d": "1/3"},
        [("nc", "NOT", ["c"])], "c",
        {"a": True, "b": True, "c": True, "nc": True, "d": True},
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "conflicts": [
            ["a"],
            ["b"],
            ["c", "nc"],
        ]
    }


def test_observed_unreachable_gate_still_supported():
    # top 不经过 g，但观测了 g：必须正常求值而非报未知引用
    body = make_body(
        {"a": "1/2", "b": "1/3"},
        [("g", "AND", ["a", "b"])], "a",
        {"g": True, "a": False},
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200, r.text
    assert r.json() == {"conflicts": [["a", "g"]]}


# ---------- 随机电路穷举对照 ----------

def test_random_circuits_against_truth_table():
    rng = random.Random(20260928)
    checked = 0
    while checked < 60:
        n = rng.randint(1, 7)
        ev_ids = [f"e{i}" for i in range(n)]
        ev_specs = {}
        for e in ev_ids:
            d = rng.randint(1, 6)
            ev_specs[e] = f"{rng.randint(0, d)}/{d}"
        gates = []
        available = list(ev_ids)
        for j in range(rng.randint(0, min(12, 3 * n))):
            gid = f"g{j}"
            if len(available) < 2:
                op = "NOT"
                inputs = [rng.choice(available)]
            else:
                op = rng.choice(["AND", "OR", "NOT"])
                if op == "NOT":
                    inputs = [rng.choice(available)]
                else:
                    k = rng.randint(2, min(6, len(available)))
                    inputs = rng.sample(available, k)
            gates.append((gid, op, inputs))
            available.append(gid)
        top = rng.choice(available)

        candidates = list(ev_ids) + [g[0] for g in gates]
        k = min(len(candidates), rng.randint(1, min(10, len(candidates))))
        chosen = rng.sample(candidates, k)
        observations = {oid: rng.choice([True, False]) for oid in chosen}

        body = make_body(ev_specs, gates, top, observations)
        r = client.post("/v1/observation-conflicts", json=body)
        expected = brute_conflicts(ev_specs, gates, top, observations)
        assert r.status_code == 200, (r.text, ev_specs, gates, observations)
        assert r.json()["conflicts"] == expected, (
            ev_specs, gates, observations, r.json(), expected
        )
        checked += 1


# ---------- 422 校验路径 ----------

def test_unknown_observation_key():
    body = make_body({"a": "1/2"}, [], "a", {"a": True, "nope": False})
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "observations" in detail and "nope" in detail


def test_non_boolean_observation_value_rejected():
    for bad in (1, 0, "true", None, 1.0):
        body = make_body({"a": "1/2"}, [], "a", {"a": bad})
        r = client.post("/v1/observation-conflicts", json=body)
        assert r.status_code == 422, bad


def test_missing_or_empty_observations_rejected():
    r = client.post("/v1/observation-conflicts", json={
        "events": [{"id": "a", "p": "1/2"}], "gates": [], "top": "a"})
    assert r.status_code == 422
    r = client.post("/v1/observation-conflicts",
                    json=make_body({"a": "1/2"}, [], "a", {}))
    assert r.status_code == 422


def test_too_many_observations_rejected():
    ev_specs = {f"e{i}": "1/2" for i in range(17)}
    body = make_body(
        ev_specs, [], "e0", {f"e{i}": True for i in range(17)}
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 422


def test_sixteen_observations_allowed():
    ev_specs = {f"e{i}": "1/2" for i in range(16)}
    body = make_body(
        ev_specs, [], "e0", {f"e{i}": True for i in range(16)}
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["conflicts"] == []


def test_extra_field_rejected():
    r = client.post("/v1/observation-conflicts", json={
        "events": [{"id": "a", "p": "1/2"}],
        "gates": [], "top": "a",
        "observations": {"a": True}, "extra": 1})
    assert r.status_code == 422


def test_original_validation_still_applies():
    # 未知 top
    body = make_body({"a": "1/2"}, [], "nope", {"a": True})
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 422 and "顶层" in r.json()["detail"]
    # 环
    body = make_body(
        {"a": "1/2"},
        [("g", "OR", ["a", "g"])], "g", {"g": True})
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 422 and "环" in r.json()["detail"]
    # 非法概率
    body = make_body({"a": "3/2"}, [], "a", {"a": True})
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 422 and "概率" in r.json()["detail"]


def test_conflict_count_limit(monkeypatch):
    # 调小组数上限：4 个独立零概率单点超过上限 3，返回说明原因的 422
    import app.conflicts as conflicts_mod
    monkeypatch.setattr(conflicts_mod, "MAX_CONFLICTS", 3)
    ev_specs = {f"e{i}": "0/1" for i in range(4)}
    body = make_body(
        ev_specs, [], "e0", {f"e{i}": True for i in range(4)}
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "5000" in detail or "冲突" in detail


def test_node_limit_returns_422(monkeypatch):
    # 调小 ROBDD 上限：合取展开新增节点超限即 422
    import app.main as main_mod
    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 5)
    evs = {f"e{i}": "1/2" for i in range(8)}
    gates = []
    prev = "e0"
    for i in range(1, 8):
        o, a, n, x = f"o{i}", f"a{i}", f"n{i}", f"x{i}"
        gates.append((o, "OR", [prev, f"e{i}"]))
        gates.append((a, "AND", [prev, f"e{i}"]))
        gates.append((n, "NOT", [a]))
        gates.append((x, "AND", [o, n]))
        prev = x
    body = make_body(
        evs, gates, prev,
        {prev: True, "e0": False, "e1": False, "e2": True},
    )
    r = client.post("/v1/observation-conflicts", json=body)
    assert r.status_code == 422
    assert "非终端节点" in r.json()["detail"]
