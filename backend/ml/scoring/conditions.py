"""条件表达式安全解析.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.4b
依据: dev-docs/complex-features/scoring-card-schema.md §3

安全策略:
    1. 禁止关键字: import / exec / eval / __ / open / file / 等
    2. 受限全局命名空间: 仅 signals 字典 + 基础运算符
    3. 信号缺失: 表达式引用的信号若不存在，条件视为 False
"""
from __future__ import annotations

import ast
import operator as op

# 支持的二元运算符
_BIN_OPS: dict[type, object] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

# 支持的比较运算符
_CMP_OPS: dict[type, object] = {
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
    ast.Lt: op.lt,
    ast.LtE: op.le,
    ast.Gt: op.gt,
    ast.GtE: op.ge,
}

# 支持的一元运算符
_UNARY_OPS: dict[type, object] = {
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Not: op.not_,
}

# 禁止的名称（防注入）
_FORBIDDEN_NAMES = {
    "import", "exec", "eval", "compile", "open", "file", "input",
    "__import__", "__builtins__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr",
    "type", "object", "class", "self", "cls",
}


class ConditionError(ValueError):
    """条件表达式语法或安全错误。"""


def _eval_node(node: ast.AST, signals: dict) -> object:
    """递归求值 AST 节点。"""
    # 常量
    if isinstance(node, ast.Constant):
        return node.value

    # 名称（变量引用）
    if isinstance(node, ast.Name):
        name = node.id
        if name in _FORBIDDEN_NAMES:
            raise ConditionError(f"禁止使用的名称: {name}")
        if name == "True":
            return True
        if name == "False":
            return False
        if name == "None":
            return None
        # 从 signals 字典取值；缺失则返回 _Missing 标记
        if name in signals:
            return signals[name]
        raise _SignalMissing(name)

    # 二元运算
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, signals)
        right = _eval_node(node.right, signals)
        if type(node.op) not in _BIN_OPS:
            raise ConditionError(f"不支持的二元运算符: {type(node.op).__name__}")
        return _BIN_OPS[type(node.op)](left, right)

    # 比较运算
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, signals)
        result = True
        for op_node, right_node in zip(node.ops, node.comparators):
            right = _eval_node(right_node, signals)
            if type(op_node) not in _CMP_OPS:
                raise ConditionError(f"不支持的比较运算符: {type(op_node).__name__}")
            result = result and _CMP_OPS[type(op_node)](left, right)
            left = right
        return result

    # 一元运算
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, signals)
        if type(node.op) not in _UNARY_OPS:
            raise ConditionError(f"不支持的一元运算符: {type(node.op).__name__}")
        return _UNARY_OPS[type(node.op)](operand)

    # 布尔运算
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, signals) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise ConditionError(f"不支持的布尔运算符: {type(node.op).__name__}")

    raise ConditionError(f"不支持的 AST 节点类型: {type(node).__name__}")


class _SignalMissing(Exception):
    """表达式引用的信号不存在。"""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"信号缺失: {name}")


def evaluate_condition(expression: str, signals: dict) -> bool:
    """安全求值条件表达式。

    Args:
        expression: 条件表达式，如 "approach_latency < 1.0 and approach_speed > 2.0"
        signals: 信号字典

    Returns:
        bool: 表达式求值结果

    Raises:
        ConditionError: 表达式语法错误或包含禁止内容
    """
    expression = expression.strip()
    if not expression:
        raise ConditionError("空表达式")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ConditionError(f"语法错误: {e}") from e

    try:
        result = _eval_node(tree.body, signals)
    except _SignalMissing:
        # 信号缺失 → 条件不命中
        return False
    except ConditionError:
        raise
    except Exception as e:
        raise ConditionError(f"求值失败: {e}") from e

    return bool(result)


def validate_expression(expression: str) -> list[str]:
    """静态检查表达式引用了哪些信号名（用于评分卡加载时校验）。

    Args:
        expression: 条件表达式

    Returns:
        表达式中引用的信号名列表（不含 True/False/None）

    Raises:
        ConditionError: 表达式语法错误或包含禁止内容
    """
    expression = expression.strip()
    if not expression:
        raise ConditionError("空表达式")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ConditionError(f"语法错误: {e}") from e

    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name = node.id
            if name in _FORBIDDEN_NAMES:
                raise ConditionError(f"禁止使用的名称: {name}")
            if name not in ("True", "False", "None") and name not in names:
                names.append(name)
    return names
