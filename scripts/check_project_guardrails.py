#!/usr/bin/env python3
"""项目守卫脚本 — 检查 truth 文档一致性 + 阶段状态 + ADR 索引完整性。

依据 AGENTS.md §6 验证命令映射 + §8 防漂移规则，在 Truth 变更时执行：
    python scripts/check_project_guardrails.py . --mode bootstrap

检查项（bootstrap 模式）：
  1. Truth 文档存在性（project-brief / function-list / technical-selection /
     architecture / runtime / stage-plan / README + stages/phase-{0..3}）
  2. ADR 完整性（decisions/0001-0010，0004 占位未触发属设计预期）
  3. 已完成阶段验收报告存在性（phase-{0,1,2}-validation.md）
  4. AGENTS.md 存在且含版本号声明
  5. 阶段状态粗一致性（AGENTS.md §7 / stage-plan.md §1 / phase-N.md 头部）
  6. 修订历史表完整性（stage-plan.md 头部声明版本 ≤ 表格最大版本）

退出码：0 = 全部通过；1 = 存在 FAIL。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- 检查项配置 ---------------------------------------------------------------

# AGENTS.md §2.1 Truth 文档 Owner（database-design / backend-boundary 未建立，
# README.md 已说明，故不列入必需）
REQUIRED_TRUTH_DOCS: list[str] = [
    "dev-docs/project-brief.md",
    "dev-docs/function-list.md",
    "dev-docs/technical-selection.md",
    "dev-docs/architecture.md",
    "dev-docs/runtime.md",
    "dev-docs/stage-plan.md",
    "dev-docs/README.md",
    "AGENTS.md",
]

# 阶段 truth 文档
REQUIRED_STAGE_DOCS: list[str] = [
    "dev-docs/stages/phase-0.md",
    "dev-docs/stages/phase-1.md",
    "dev-docs/stages/phase-2.md",
    "dev-docs/stages/phase-3.md",
]

# ADR 0001-0010（0004 占位预留给 PoseC3D 触发但失败记录，未触发创建属设计预期）
EXPECTED_ADRS: list[int] = [1, 2, 3, 5, 6, 7, 8, 9, 10]

# 已完成阶段（AGENTS.md §7 标记 ✅ 完成的阶段）必须有验收报告
COMPLETED_PHASES: list[int] = [0, 1, 2]


# --- 检查结果模型 -------------------------------------------------------------

@dataclass
class CheckResult:
    """单项检查结果。"""
    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    detail: str


@dataclass
class GuardrailReport:
    """守卫报告汇总。"""
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str) -> None:
        self.results.append(CheckResult(name, status, detail))

    @property
    def has_fail(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)

    def summary(self) -> str:
        n_pass = sum(1 for r in self.results if r.status == "PASS")
        n_warn = sum(1 for r in self.results if r.status == "WARN")
        n_fail = sum(1 for r in self.results if r.status == "FAIL")
        return f"{n_pass} PASS / {n_warn} WARN / {n_fail} FAIL"


# --- 检查函数 -----------------------------------------------------------------

def check_truth_docs_exist(root: Path) -> tuple[str, str]:
    """检查必需 truth 文档存在性。"""
    missing: list[str] = []
    for rel in REQUIRED_TRUTH_DOCS + REQUIRED_STAGE_DOCS:
        if not (root / rel).is_file():
            missing.append(rel)
    if missing:
        return "FAIL", f"缺失 truth 文档: {', '.join(missing)}"
    return "PASS", f"全部 {len(REQUIRED_TRUTH_DOCS) + len(REQUIRED_STAGE_DOCS)} 份 truth 文档存在"


def check_adr_completeness(root: Path) -> tuple[str, str]:
    """检查 ADR 0001-0010 完整性（0004 占位除外）。"""
    decisions_dir = root / "dev-docs" / "decisions"
    if not decisions_dir.is_dir():
        return "FAIL", "dev-docs/decisions/ 目录不存在"
    existing = {f.stem.split("-", 1)[0] for f in decisions_dir.glob("*.md") if f.is_file()}
    missing = [f"{n:04d}" for n in EXPECTED_ADRS if f"{n:04d}" not in existing]
    # 0004 占位未触发属设计预期，单独提示
    adr0004_exists = "0004" in existing
    if missing:
        return "FAIL", f"缺失 ADR: {', '.join(missing)}"
    note = "（0004 占位未触发，符合设计预期）" if not adr0004_exists else ""
    return "PASS", f"ADR {min(EXPECTED_ADRS):04d}-{max(EXPECTED_ADRS):04d} 齐全{note}"


def check_validation_reports(root: Path) -> tuple[str, str]:
    """检查已完成阶段的验收报告存在性。"""
    reports_dir = root / "reports"
    if not reports_dir.is_dir():
        return "FAIL", "reports/ 目录不存在"
    missing: list[str] = []
    for phase in COMPLETED_PHASES:
        report = reports_dir / f"phase-{phase}-validation.md"
        if not report.is_file():
            missing.append(report.name)
    if missing:
        return "FAIL", f"缺失阶段验收报告: {', '.join(missing)}"
    return "PASS", f"已完成阶段 {COMPLETED_PHASES} 验收报告齐全"


def check_agents_md_version(root: Path) -> tuple[str, str]:
    """检查 AGENTS.md 存在且含版本号声明。"""
    agents = root / "AGENTS.md"
    if not agents.is_file():
        return "FAIL", "AGENTS.md 不存在"
    text = agents.read_text(encoding="utf-8")
    m = re.search(r"状态:\s*✅\s*v(\d+\.\d+)", text)
    if not m:
        return "WARN", "AGENTS.md 未找到 '状态: ✅ vX.Y' 版本声明"
    return "PASS", f"AGENTS.md 版本 v{m.group(1)}"


def check_stage_status_consistency(root: Path) -> tuple[str, str]:
    """检查阶段状态粗一致性（AGENTS.md §7 / stage-plan.md §1）。

    仅校验已完成阶段（0/1/2）在两处均标记 ✅，Phase 3 标记 🔄 实施中或 ✅ 完成。
    """
    issues: list[str] = []

    agents_text = (root / "AGENTS.md").read_text(encoding="utf-8")
    plan_text = (root / "dev-docs" / "stage-plan.md").read_text(encoding="utf-8")

    # AGENTS.md §7 阶段映射表行：| Phase N ... | ✅ 完成 ... |
    for phase in COMPLETED_PHASES:
        # 在 AGENTS.md 中查找 Phase N 行是否含 ✅
        pattern_agents = rf"\|\s*Phase {phase}\b.*✅"
        if not re.search(pattern_agents, agents_text):
            issues.append(f"AGENTS.md Phase {phase} 未标记 ✅")
        # 在 stage-plan.md 中查找 Phase N 行是否含 ✅
        pattern_plan = rf"\|\s*Phase {phase}\b.*✅"
        if not re.search(pattern_plan, plan_text):
            issues.append(f"stage-plan.md Phase {phase} 未标记 ✅")

    # Phase 3 应标记 🔄 实施中
    if not re.search(r"\|\s*Phase 3\b.*(🔄|✅)", agents_text):
        issues.append("AGENTS.md Phase 3 未标记 🔄/✅")
    if not re.search(r"\|\s*Phase 3\b.*(🔄|✅)", plan_text):
        issues.append("stage-plan.md Phase 3 未标记 🔄/✅")

    if issues:
        return "FAIL", "；".join(issues)
    return "PASS", "AGENTS.md §7 与 stage-plan.md §1 阶段状态一致（Phase 0/1/2 ✅，Phase 3 🔄/✅）"


def check_stage_plan_revision_history(root: Path) -> tuple[str, str]:
    """检查 stage-plan.md 修订历史表完整性（头部声明版本 ≤ 表格最大版本）。

    防止文档漂移：头部声明 v1.4 但表格只到 v1.2 这类问题。
    """
    plan = root / "dev-docs" / "stage-plan.md"
    if not plan.is_file():
        return "FAIL", "stage-plan.md 不存在"
    text = plan.read_text(encoding="utf-8")

    # 头部声明版本：提取 "v1.4" 这类（取最后一个版本号）
    header_match = re.search(r"状态:.*?v(\d+\.\d+)", text[:500])
    if not header_match:
        return "WARN", "stage-plan.md 头部未找到版本声明"
    header_ver = header_match.group(1)

    # 修订历史表中的版本号：| v1.X | 日期 | ... |
    table_versions = re.findall(r"^\|\s*v(\d+\.\d+)\s*\|", text, re.MULTILINE)
    if not table_versions:
        return "WARN", "stage-plan.md 修订历史表无版本条目"

    max_table_ver = table_versions[-1]  # 表格最后一条即最新
    if header_ver != max_table_ver:
        return "FAIL", (
            f"stage-plan.md 版本漂移：头部声明 v{header_ver}，"
            f"但修订历史表最新为 v{max_table_ver}"
        )
    return "PASS", f"stage-plan.md 修订历史完整（v{header_ver}）"


def check_phase_doc_status_header(root: Path) -> tuple[str, str]:
    """检查 phase-N.md 头部状态行存在且非空。"""
    issues: list[str] = []
    for rel in REQUIRED_STAGE_DOCS:
        doc = root / rel
        if not doc.is_file():
            issues.append(f"{rel} 不存在")
            continue
        head = doc.read_text(encoding="utf-8")[:800]
        if not re.search(r"状态:", head):
            issues.append(f"{rel} 头部缺 '状态:' 声明")
    if issues:
        return "FAIL", "；".join(issues)
    return "PASS", "全部阶段文档头部状态声明存在"


# --- 主流程 -------------------------------------------------------------------

def run_guardrails(root: Path, mode: str) -> GuardrailReport:
    """执行守卫检查，返回报告。"""
    report = GuardrailReport()

    checks: list[tuple[str, str]] = [
        ("Truth 文档存在性", "check_truth_docs_exist"),
        ("ADR 完整性", "check_adr_completeness"),
        ("已完成阶段验收报告", "check_validation_reports"),
        ("AGENTS.md 版本声明", "check_agents_md_version"),
        ("阶段状态一致性", "check_stage_status_consistency"),
        ("stage-plan.md 修订历史", "check_stage_plan_revision_history"),
        ("阶段文档状态头部", "check_phase_doc_status_header"),
    ]

    if mode == "bootstrap":
        for name, func_name in checks:
            func = globals()[func_name]
            status, detail = func(root)
            report.add(name, status, detail)
    else:
        report.add("模式", "WARN", f"未知模式 '{mode}'，仅支持 bootstrap")

    return report


def print_report(report: GuardrailReport) -> None:
    """打印守卫报告。"""
    print("=" * 72)
    print("项目守卫检查报告")
    print("=" * 72)
    for r in report.results:
        symbol = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[r.status]
        print(f"{symbol} [{r.status:4}] {r.name}")
        print(f"          {r.detail}")
    print("-" * 72)
    print(f"汇总: {report.summary()}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="项目守卫脚本 — 检查 truth 文档一致性 + 阶段状态 + ADR 索引完整性",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="项目根目录路径（如 . 或 d:\\Desktop\\k9-training-system）",
    )
    parser.add_argument(
        "--mode",
        default="bootstrap",
        choices=["bootstrap"],
        help="检查模式（默认 bootstrap，Truth 变更时执行）",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"错误: 项目根目录不存在: {root}", file=sys.stderr)
        return 2

    report = run_guardrails(root, args.mode)
    print_report(report)
    return 1 if report.has_fail else 0


if __name__ == "__main__":
    sys.exit(main())
