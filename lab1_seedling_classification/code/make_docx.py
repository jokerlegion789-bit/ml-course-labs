# -*- coding: utf-8 -*-
"""把 reports/实验报告.md 转成 Word 版（含表格与插图）。

用法: python make_docx.py [输入.md] [输出.docx]
支持：#/##/### 标题、段落、无序列表、表格、``` 代码块、**加粗**、行内 `代码`，
并在文末按文件名顺序插入 reports/fig*.png 配图（若正文未引用）。
"""
import os
import re
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

import common as C


def set_base_style(doc):
    st = doc.styles["Normal"]
    st.font.name = "宋体"
    st.font.size = Pt(10.5)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    for i, sz in ((1, 16), (2, 14), (3, 12)):
        h = doc.styles[f"Heading {i}"]
        h.font.name = "黑体"
        h.font.size = Pt(sz)
        h.font.color.rgb = RGBColor(0, 0, 0)
        try:
            h._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        except Exception:
            pass


def add_rich(p, text):
    """处理 **加粗** 与 `代码`。"""
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = p.add_run(part[1:-1]); r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        else:
            p.add_run(part)


def convert(md_path, docx_path, embed_figs=True):
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    doc = Document()
    set_base_style(doc)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Cm(2.54)
        s.left_margin = s.right_margin = Cm(2.8)

    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.6)
            r = p.add_run("\n".join(buf))
            r.font.name = "Consolas"; r.font.size = Pt(9)
            continue
        if s.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not set("".join(cells)) <= set("-: "):
                    rows.append(cells)
                i += 1
            if rows:
                ncol = max(len(r) for r in rows)
                t = doc.add_table(rows=0, cols=ncol)
                t.style = "Light Grid Accent 1"
                for ri, rw in enumerate(rows):
                    cells = t.add_row().cells
                    for ci in range(ncol):
                        txt = rw[ci] if ci < len(rw) else ""
                        cells[ci].text = ""
                        p = cells[ci].paragraphs[0]
                        add_rich(p, txt)
                        for run in p.runs:
                            run.font.size = Pt(9)
                            if ri == 0:
                                run.bold = True
                doc.add_paragraph()
            continue
        if not s:
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = min(len(m.group(1)), 4)
            doc.add_heading(m.group(2), level=lvl)
            i += 1
            continue
        if s.startswith("---"):
            i += 1
            continue
        if re.match(r"^[-*]\s+", s):
            p = doc.add_paragraph(style="List Bullet")
            add_rich(p, re.sub(r"^[-*]\s+", "", s))
            i += 1
            continue
        if re.match(r"^\d+\.\s+", s):
            p = doc.add_paragraph(style="List Number")
            add_rich(p, re.sub(r"^\d+\.\s+", "", s))
            i += 1
            continue
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74)
        add_rich(p, s)
        i += 1

    if embed_figs:
        figs = sorted(f for f in os.listdir(C.REPORTS)
                      if f.startswith("fig") and f.endswith(".png"))
        if figs:
            doc.add_page_break()
            doc.add_heading("附：实验配图", level=1)
            for fn in figs:
                doc.add_paragraph(fn.replace(".png", ""))
                try:
                    doc.add_picture(os.path.join(C.REPORTS, fn), width=Cm(16))
                    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                except Exception as e:
                    doc.add_paragraph(f"（插图失败：{e}）")
    doc.save(docx_path)
    print(f"  [Word] {docx_path}")


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.REPORTS, "实验报告.md")
    dx = sys.argv[2] if len(sys.argv) > 2 else os.path.join(C.REPORTS, "实验报告.docx")
    convert(md, dx)
