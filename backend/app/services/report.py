"""PDF 评分报告生成.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 1.4c
依据: dev-docs/stages/phase-1.md §1.4c

功能:
    1. 接收犬只信息 + 视频信息 + ScoringResult + 行为时间线
    2. 生成 PDF 报告（reportlab platypus）
    3. 输出到 reports/{video_id}.pdf

报告结构:
    1. 标题 + 元信息（生成时间、场景、版本）
    2. 犬只信息表
    3. 视频信息表
    4. 评分摘要（总分 + 判定）
    5. 评分明细表（维度 + 得分 + 标签 + 命中规则）
    6. 行为时间线（行为类别 + 起止时间 + 置信度）
    7. 评分说明（人类可读解释）

用法:
    from backend.app.services.report import ReportInput, generate_report
    from backend.ml.scoring import ScoringResult

    report_input = ReportInput(
        dog_name="雷克斯",
        breed="比利时马犬",
        video_filename="test.mp4",
        video_duration_sec=30.0,
        scoring_result=result,  # ScoringResult
        behaviors=episodes,     # list[BehaviorEpisode]
    )
    pdf_path = generate_report(report_input, output_path="reports/123.pdf")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from backend.ml.behavior.rule_engine import BehaviorEpisode
from backend.ml.scoring import ScoringResult

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================
# 数据契约
# ============================================================


@dataclass
class ReportInput:
    """PDF 报告输入数据。"""

    # 犬只信息
    dog_name: str
    breed: Optional[str] = None
    birth_date: Optional[str] = None  # ISO 格式字符串
    gender: Optional[str] = None  # male / female
    chip_id: Optional[str] = None
    training_stage: Optional[str] = None  # P0 / P1 / P2

    # 视频信息
    video_filename: str = ""
    video_duration_sec: Optional[float] = None
    video_fps: Optional[float] = None
    video_resolution: Optional[str] = None  # "1920x1080"
    uploaded_at: Optional[str] = None  # ISO 格式字符串

    # 评分结果
    scoring_result: ScoringResult = None  # type: ignore[assignment]

    # 行为时间线
    behaviors: list[BehaviorEpisode] = field(default_factory=list)

    # 元数据
    video_id: Optional[int] = None
    handler_name: Optional[str] = None
    notes: Optional[str] = None


# ============================================================
# 样式
# ============================================================


def _build_styles() -> dict[str, ParagraphStyle]:
    """构建段落样式。"""
    styles = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#1a3a52"),
            spaceAfter=6 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.grey,
            alignment=1,  # center
            spaceAfter=10 * mm,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#1a3a52"),
            spaceBefore=8 * mm,
            spaceAfter=4 * mm,
        ),
        "normal": ParagraphStyle(
            "Normal",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
        ),
        "verdict_pass": ParagraphStyle(
            "VerdictPass",
            parent=styles["Normal"],
            fontSize=16,
            textColor=colors.HexColor("#2e7d32"),
            alignment=1,
            fontName="Helvetica-Bold",
        ),
        "verdict_borderline": ParagraphStyle(
            "VerdictBorderline",
            parent=styles["Normal"],
            fontSize=16,
            textColor=colors.HexColor("#f57c00"),
            alignment=1,
            fontName="Helvetica-Bold",
        ),
        "verdict_fail": ParagraphStyle(
            "VerdictFail",
            parent=styles["Normal"],
            fontSize=16,
            textColor=colors.HexColor("#c62828"),
            alignment=1,
            fontName="Helvetica-Bold",
        ),
    }


# 判定中文映射
_VERDICT_CN = {
    "pass": "合格",
    "borderline": "基本合格",
    "fail": "不合格",
}

# 性别中文映射
_GENDER_CN = {"male": "公", "female": "母"}


# ============================================================
# 报告生成
# ============================================================


def generate_report(
    report_input: ReportInput,
    output_path: Optional[str | Path] = None,
) -> Path:
    """生成 PDF 评分报告。

    Args:
        report_input: 报告输入数据
        output_path: 输出路径（默认 reports/{video_id}.pdf 或 reports/report.pdf）

    Returns:
        PDF 文件路径
    """
    if output_path is None:
        vid = report_input.video_id or "report"
        output_path = REPORTS_DIR / f"{vid}.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 构建文档
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=f"工作犬评分报告 - {report_input.dog_name}",
    )

    styles = _build_styles()
    story: list[Any] = []

    # === 1. 标题 ===
    scene_cn = "幼犬选育" if report_input.scoring_result.scene == "puppy_selection" else "科目测评"
    story.append(Paragraph(f"工作犬{scene_cn}评分报告", styles["title"]))
    story.append(
        Paragraph(
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"评分卡版本: {report_input.scoring_result.card_version}",
            styles["subtitle"],
        )
    )

    # === 2. 犬只信息 ===
    story.append(Paragraph("犬只信息", styles["h2"]))
    dog_data = [
        ["犬名", report_input.dog_name, "品种", report_input.breed or "—"],
        ["性别", _GENDER_CN.get(report_input.gender, "—"), "出生日期", report_input.birth_date or "—"],
        ["芯片号", report_input.chip_id or "—", "训练阶段", report_input.training_stage or "—"],
    ]
    if report_input.handler_name:
        dog_data.append(["训导员", report_input.handler_name, "备注", report_input.notes or "—"])
    dog_table = Table(dog_data, colWidths=[2.5 * cm, 5 * cm, 2.5 * cm, 6 * cm])
    dog_table.setStyle(_info_table_style())
    story.append(dog_table)

    # === 3. 视频信息 ===
    if report_input.video_filename:
        story.append(Paragraph("视频信息", styles["h2"]))
        duration_str = (
            f"{report_input.video_duration_sec:.1f} 秒"
            if report_input.video_duration_sec
            else "—"
        )
        fps_str = f"{report_input.video_fps:.1f} fps" if report_input.video_fps else "—"
        video_data = [
            ["文件名", report_input.video_filename, "时长", duration_str],
            ["帧率", fps_str, "分辨率", report_input.video_resolution or "—"],
        ]
        if report_input.uploaded_at:
            video_data.append(["上传时间", report_input.uploaded_at, "", ""])
        video_table = Table(video_data, colWidths=[2.5 * cm, 5 * cm, 2.5 * cm, 6 * cm])
        video_table.setStyle(_info_table_style())
        story.append(video_table)

    # === 4. 评分摘要 ===
    story.append(Paragraph("评分摘要", styles["h2"]))
    sr = report_input.scoring_result
    verdict_cn = _VERDICT_CN.get(sr.verdict, sr.verdict)
    verdict_style = styles[f"verdict_{sr.verdict}"]

    story.append(Paragraph(f"总分: {sr.total_score:.1f} / 100", styles["normal"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(f"判定: {verdict_cn}", verdict_style))
    story.append(Spacer(1, 5 * mm))

    # === 5. 评分明细表 ===
    story.append(Paragraph("评分明细", styles["h2"]))
    detail_header = ["维度", "权重", "得分", "标签", "命中规则"]
    detail_rows = [detail_header]
    for d in sr.dimension_details:
        detail_rows.append(
            [
                d.dimension_name,
                f"{d.weight:.0%}",
                str(d.score),
                d.label,
                d.hit_rule_id or "默认",
            ]
        )
    detail_table = Table(detail_rows, colWidths=[3 * cm, 2 * cm, 2 * cm, 3 * cm, 5 * cm])
    detail_table.setStyle(_detail_table_style())
    story.append(detail_table)

    # === 6. 行为时间线 ===
    if report_input.behaviors:
        story.append(Paragraph("行为时间线", styles["h2"]))
        beh_header = ["行为类别", "起始时间", "结束时间", "持续(秒)", "置信度"]
        beh_rows = [beh_header]
        for b in report_input.behaviors:
            start_sec = b.start_frame / 30.0 if b.start_frame else 0.0  # 假设 30fps
            end_sec = b.end_frame / 30.0 if b.end_frame else 0.0
            duration = end_sec - start_sec
            beh_rows.append(
                [
                    b.behavior,
                    f"{start_sec:.2f}s",
                    f"{end_sec:.2f}s",
                    f"{duration:.2f}",
                    f"{b.confidence:.2f}",
                ]
            )
        beh_table = Table(beh_rows, colWidths=[3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 2.5 * cm])
        beh_table.setStyle(_detail_table_style())
        story.append(beh_table)

    # === 7. 评分说明 ===
    if sr.explanation:
        story.append(Paragraph("评分说明", styles["h2"]))
        for line in sr.explanation:
            story.append(Paragraph(f"• {line}", styles["normal"]))

    # 构建 PDF
    doc.build(story)
    return output_path


# ============================================================
# 表格样式辅助
# ============================================================


def _info_table_style() -> TableStyle:
    """信息表样式（无表头，键值对）。"""
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#666666")),  # 键列灰色
            ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#666666")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),  # 键列加粗
            ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ]
    )


def _detail_table_style() -> TableStyle:
    """明细表样式（带表头）。"""
    return TableStyle(
        [
            # 表头
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a52")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            # 数据行
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("ALIGN", (0, 1), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            # 斑马纹
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            # 边框
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ]
    )
