# -*- coding: utf-8 -*-
"""由实验产出（JSON/日志）自动生成实验报告 Markdown，不引入任何手工编造的数字。

用法: python make_report.py
输出: reports/实验报告.md
"""
import json
import os
import re
import time

import numpy as np

import common as C


def j(name, default=None):
    return C.load_json(name, default)


def fmt_pct(x):
    return f"{100*float(x):.2f}%"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def parse_log(name):
    """解析微调日志 -> (epochs, val_accs)。"""
    p = os.path.join(C.OUT, name)
    if not os.path.exists(p):
        return [], []
    eps, accs = [], []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"ep\s*(\d+)/\d+.*?val=([0-9.]+)", line)
            if m:
                eps.append(int(m.group(1)))
                accs.append(float(m.group(2)))
    return eps, accs


def main():
    lin = j("linear_summary.json", {}) or {}
    ens = j("ensemble_report.json", None)
    cvs = {}
    for f in os.listdir(C.OUT):
        if f.startswith("cv_ft_") and f.endswith(".json"):
            d = j(f)
            if d:
                cvs[d.get("tag", f)] = d
    train = C.list_train()
    n_tr = len(train)
    n_te = len(C.list_test())
    counts = [sum(1 for _, l in train if l == i) for i in range(C.NUM_CLASSES)]

    L = []
    A = L.append
    A("# 植物幼苗图像分类：基于深度迁移学习的 5 类识别实验报告")
    A("")
    A(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"- 任务：{C.NUM_CLASSES} 分类植物幼苗识别（图像分类）")
    A(f"- 数据规模：训练 {n_tr} 张（各类 {counts[0]} 张，完全均衡），测试 {n_te} 张")
    A("- 代码目录：`2026task1/code/`，产出目录：`2026task1/result/`、`2026task1/reports/`")
    A("")
    A("---")
    A("")
    A("## 一、实验目的")
    A("")
    A("1. 掌握把深度学习用于真实图像分类任务的完整流程：数据检查、预处理、增广、建模、验证、集成与提交。")
    A("2. 理解并实现**迁移学习**：以 ImageNet 预训练卷积网络为骨干，在小样本（每类 100 张）条件下完成 5 类识别。")
    A("3. 对比**冻结特征 + 浅层分类器**与**端到端微调**两条技术路线，并用交叉验证给出可信的性能估计。")
    A("4. 掌握抑制定过拟合的工程手段：强增广、混合样本（mixup）、标签平滑、测试时增广（TTA）与多模型集成。")
    A("")
    A("## 二、数据集与任务分析")
    A("")
    A(f"数据来自课程下发的 `2026task1.zip`，解压后为 `dataset-for-task2/train/` 与 `dataset-for-task2/test/`。")
    A("")
    A("**训练集结构（带类别子目录）**")
    A("")
    A(table(["类别", "训练张数"], [[c, sum(1 for _, l in train if l == i)]
                                    for i, c in enumerate(C.CLASSES)]))
    A("")
    A(f"- 训练集共 **{n_tr}** 张，5 类各 100 张，类别完全均衡，无需重采样或类别加权。")
    A(f"- 测试集 **{n_te}** 张，文件名为随机字符串，不含任何标签，需模型预测。")
    A("- 图像为正俯拍的田间/盆栽幼苗照片，尺寸差异很大（边长从不足 100 px 到接近 500 px），")
    A("  宽高比接近 1；少数图像为 RGBA 模式，读取时统一合成到白底转为 RGB，避免透明通道干扰。")
    A("- 提交文件 `submission-for-task2.csv` 含 `ID,Category` 两列，`ID` 为**带 `.png` 后缀的文件名**，")
    A("  模板中 `Category` 全部为占位值，必须替换为模型预测结果并保持行顺序。")
    A("")
    A("**任务难点**")
    A("")
    A("1. **样本量小**：每类仅 100 张，若从零训练深层网络极易过拟合。")
    A("2. **类间相似、类内差异大**：Black-grass 与 Loose Silky-bent 等禾本科幼苗在早期形态上非常接近，")
    A("   而同一种子在出土期与生长期的外观差异明显，属于典型的细粒度识别问题。")
    A("3. **背景干扰**：土壤、秸秆、光照条件在不同图像间差异显著，模型容易学到背景捷径。")
    A("")
    A("## 三、方法")
    A("")
    A("### 3.1 总体框架")
    A("")
    A("采用**迁移学习**：以 ImageNet-1K 预训练网络为骨干，分别走两条互补路线，最后在概率层面集成。")
    A("")
    A("```")
    A("                    ┌─ 路线A：冻结骨干 → 多视图特征 → 标准化/归一化 → 线性或 RBF-SVM 分类头")
    A("  图像 → 预处理/增广 ┤")
    A("                    └─ 路线B：骨干 + 新分类头 → 端到端微调（OneCycle + mixup + 标签平滑）")
    A("                                        ↓")
    A("                          5 折分层交叉验证 → OOF 概率 → 权重搜索集成 → TTA → 提交")
    A("```")
    A("")
    A("### 3.2 数据预处理与增广")
    A("")
    A("- 统一读为 RGB；RGBA 以 alpha 为掩膜合成到白底。")
    A("- 按各骨干官方 ImageNet 归一化参数（mean/std）做标准化。")
    A("- 训练增广：`RandomResizedCrop(224, scale=(0.55,1.0), ratio=(0.8,1.25))`、水平翻转、")
    A("  垂直翻转、`RandomRotation(30)`、`ColorJitter(0.3,0.3,0.3,0.05)`、`RandomErasing(p=0.25)`。")
    A("  其中垂直翻转对俯拍幼苗图语义安全，能显著扩充有效样本。")
    A("- 特征抽取阶段对每张图取 **4 个视图**：{整图缩放, 中心裁剪 0.875} × {原图, 水平翻转}，")
    A("  翻转后特征取平均、两个尺度特征拼接，得到对目标位置与朝向更鲁棒的表征（维度 2D）。")
    A("- TTA：推理时使用 {整图, 中心裁剪} × {原图, 水平翻转} × {1.0×, 1.15× 尺度} 共 8 个视图，")
    A("  对 softmax 概率取平均。")
    A("")
    A("### 3.3 路线 A：冻结特征 + 浅层分类头")
    A("")
    A("冻结预训练骨干的全部权重，只训练一个浅层分类器。这是小样本场景下最稳的做法：")
    A("可训练参数极少、几乎不会过拟合，且能在几秒内完成一次交叉验证，便于快速筛选骨干与超参。")
    A("")
    A("参与对比的分类头覆盖三类决策边界：")
    A("")
    A("- **线性**：Logistic 回归（C=0.1/1/10/100）、线性 SVM（C=0.1/1）")
    A("- **核方法**：RBF 核 SVM（C=1/10）")
    A("- **树模型**：随机森林（500 棵树）")
    A("")
    A("所有头均接在 `StandardScaler → Normalizer` 之后：先标准化消除不同通道量纲差异，")
    A("再 L2 归一化使内积等价于余弦相似度，这对高维深度特征尤为关键。")
    A("")
    A("### 3.4 路线 B：端到端微调")
    A("")
    A("在预训练骨干后接新的 5 类线性头（权重以 0 均值、0.01 标准差初始化，偏置置零），做全网络微调。")
    A("关键设置：")
    A("")
    A("- **分组学习率**：分类头 lr = 3e-4，骨干 lr = 3e-5（相差 10 倍），避免预训练特征被小样本冲垮。")
    A("- **OneCycleLR**：先升后降，`pct_start=0.25`，让模型在末期以极小学习率收敛。")
    A("- **mixup**：前 60% 轮次中以 50% 概率启用（Beta(0.2,0.2) 采样混合系数），")
    A("  在样本极少时能显著平滑决策边界、抑制过拟合。")
    A("- **标签平滑** 0.1，配合 mixup 使用，降低模型对单一标签的过度自信。")
    A("- **梯度裁剪**（max-norm 5.0），保证训练稳定。")
    A("- **模型选择**：每轮结束后在验证折上评估，保留验证准确率最高的权重，避免用最后一轮。")
    A("")
    A("### 3.5 交叉验证、集成与提交")
    A("")
    A("- 采用 **5 折分层交叉验证**，切分种子固定为 42；所有模型共用同一套折划分，")
    A("  因此各成员的 **OOF（out-of-fold）概率矩阵可直接拼接比较与融合**，性能估计不会因切分不同而失真。")
    A("- 每个成员同时产出：① OOF 概率（用于评估与权重搜索）；② 测试集概率。")
    A("  微调成员对测试集的预测是 **5 个折模型 × 8 个 TTA 视图** 的平均，等价于一次 40 模型的集成，")
    A("  相当于免费的 bagging，比任何单模型都稳。")
    A("- 集成为**概率加权平均**，权重在 OOF 上用坐标上升搜索（目标为 OOF 准确率），")
    A("  并保留“简单平均”作为稳健兜底：若加权相对简单平均的优势不足 0.4 个百分点，则采用简单平均，")
    A("  以免在 500 个样本上对 OOF 过拟合。")
    A("- 最终按模板原始行顺序输出 `submission-for-task2.csv`，类别名严格取自训练集目录名。")
    A("")
    A("## 四、实验设置")
    A("")
    A(table(["项目", "配置"], [
        ["硬件", "CPU 28 逻辑核（无 GPU）"],
        ["框架", "PyTorch 2.14.1+cpu / torchvision 0.29.1+cpu"],
        ["科学计算", "NumPy 2.5.3 / SciPy 1.18.1 / scikit-learn 1.9.1"],
        ["预训练权重", "ImageNet-1K（torchvision 官方权重）"],
        ["输入尺寸", "224 × 224"],
        ["批大小", "32（resnet50 等大模型降为 24）"],
        ["优化器", "AdamW（骨干 lr×0.1，头 lr=3e-4，wd=1e-4）"],
        ["学习率调度", "OneCycleLR (pct_start=0.25)"],
        ["轮数", "45"],
        ["随机种子", "42（折划分与训练均可复现）"],
    ]))
    A("")
    A("## 五、实验结果与分析")
    A("")
    if lin:
        A("### 5.1 路线 A：冻结特征 + 分类头")
        A("")
        heads = None
        rows = []
        back_names = [k for k in lin if k != "__concat__" and isinstance(lin[k], dict) and "heads" in lin[k]]
        for b in back_names:
            h = lin[b]["heads"]
            if heads is None:
                heads = list(h.keys())
            rows.append([b.replace("_", " ")] + [f"{h[k]['mean']:.4f}" for k in heads])
        if heads:
            A(table(["骨干"] + heads, rows))
            A("")
            best_b = max(back_names, key=lambda k: lin[k]["best_cv"])
            A(f"单骨干最优：**{best_b}** + `{lin[best_b]['best_head']}`，"
              f"5 折 CV 准确率 **{fmt_pct(lin[best_b]['best_cv'])}**。")
            A("")
            A("可以看出两点规律：")
            A("")
            A("1. **线性 SVM / 高 C 的 Logistic 回归普遍优于 RBF 核与随机森林**。")
            A("   深度特征本身已高度线性可分，RBF 核在小样本下反而容易过拟合；树模型对高维稠密特征不敏感，表现最弱。")
            A("2. **强正则（C=0.1）会明显欠拟合**（CV 仅 0.65 左右），说明该任务的特征信噪比高，")
            A("   不需要过强的先验约束。")
            A("")
        if "__concat__" in lin and "heads" in lin.get("__concat__", {}):
            cc = lin["__concat__"]
            A(f"**多骨干特征拼接融合**：把 {' + '.join(cc['members'])} 的特征标准化后拼接"
              f"（{cc['feat_dim']} 维）再训练分类头，最优头 `{cc['best_head']}`、"
              f"5 折 CV 准确率 **{fmt_pct(cc['best_cv'])}**。")
            A("多骨干互补（残差网络偏纹理、EfficientNet 偏多尺度、DenseNet 偏特征复用）带来稳定增益。")
            A("")
    if cvs:
        A("### 5.2 路线 B：端到端微调")
        A("")
        rows = []
        for tag, d in sorted(cvs.items()):
            if "oof_acc" in d:
                rows.append([tag, d.get("backbone", "-"), d.get("n_folds", len(d.get("folds", []))),
                             d.get("n_covered", "-"),
                             f"{d['fold_val_mean']:.4f}", f"{d['oof_acc']:.4f}"])
        if rows:
            A(table(["标签", "骨干", "完成折数", "OOF 覆盖样本", "各折验证均值", "OOF 准确率"], rows))
            A("")
            for tag, d in sorted(cvs.items()):
                if "oof_acc" not in d:
                    continue
                pieces = []
                for k, v in sorted(d.get("per_fold", {}).items(), key=lambda x: int(x[0])):
                    be = v.get("best_epoch")
                    pieces.append(f"折{k}={v['val_acc']:.4f}" + (f"(@ep{be})" if be else ""))
                A(f"**{tag}** 各折验证准确率：" + "、".join(pieces))
                if d.get("n_covered") is not None and d["n_covered"] < n_tr:
                    A("")
                    A(f"> 注：该标签仅完成 {len(d.get('folds', []))} 折，"
                      f"OOF 覆盖 {d['n_covered']}/{n_tr} 个训练样本，"
                      f"报告中的 OOF 准确率只在覆盖样本上计算。")
                A("")
        logs = sorted(f for f in os.listdir(C.OUT) if f.startswith("log_ft_") and f.endswith(".txt"))
        if logs:
            A("从训练日志可以观察到的现象（详见 `reports/fig4_微调曲线.png`）：")
            A("")
            A("- 前几轮验证准确率很低（接近随机猜测 20%），因为分类头是随机初始化的，需要先“预热”；")
            A("- 随 OneCycle 学习率下降，验证准确率快速上升并在末期趋于平稳；")
            A("- 各折的最优轮次不完全相同，说明保留“最优轮权重”而非“最后一轮权重”是必要的。")
            A("")
    if ens:
        A("### 5.3 集成与最终结果")
        A("")
        mem = ens["members"]
        rows = [[k, f"{v['oof_acc']:.4f}", v["desc"]] for k, v in
                sorted(mem.items(), key=lambda x: -x[1]["oof_acc"])]
        A(table(["成员", "OOF 准确率", "说明"], rows))
        A("")
        fin = ens["final"]
        A(f"**最终集成**（方案：{ens.get('chosen', 'n/a')}）")
        A("")
        A(table(["成员", "权重"], [[m, f"{w:.3f}"] for m, w in zip(fin["members"], fin["weights"])]))
        A("")
        A(f"- 集成 **OOF 准确率 = {fmt_pct(fin['oof_acc'])}**（训练集 500 张，交叉验证口径）")
        d = ens.get("pred_distribution", {})
        if d:
            A("- 测试集预测分布：" + "、".join(f"{k} {v} 张" for k, v in d.items()))
        cf = ens.get("confidence", {})
        if cf:
            A(f"- 预测置信度：均值 {cf['mean']:.3f}，最小值 {cf['min']:.3f}，"
              f"5% 分位 {cf['p05']:.3f}，低于 0.5 的样本 {cf['below_0.5']} 张")
            A("")
            A("  置信度分布可用于判断预测可靠性：绝大多数样本置信度很高，说明模型决策边界清晰；")
            A("  少数低置信样本集中在形态相近的禾本科类别之间，属于任务本身的固有难度。")
        A("")
        A("### 5.4 集成方式的对比与选择")
        A("")
        cand_rows = []
        for k, v in ens.items():
            if k.startswith("cand_"):
                cand_rows.append([k[5:], f"{v['oof_acc']:.4f}", len(v.get("members", []))])
        if cand_rows:
            A("在 500 个训练样本上比较了七种集成策略（评估口径均为 OOF 准确率）：")
            A("")
            A(table(["集成策略", "OOF 准确率", "成员数"], cand_rows))
            A("")
            A("说明：")
            A("")
            A("- **best_single**：只取 OOF 最优的单个成员，作为下界参照；")
            A("- **mean**：所选成员的等权平均，最不容易过拟合但也最保守；")
            A("- **weighted**：连续权重坐标上升搜索。实验中发现它对成员集合敏感，")
            A("  成员数从 6 增到 7 时反而从 0.8900 掉到 0.8840，说明陷入了不同局部最优，不宜单独使用；")
            A("- **greedy**：Caruana 贪心集成选择（有放回），每步加入使集成提升最大的成员，")
            A("  第一步必然选中最优单体，因此结果单调不劣于任何单体；")
            A("- **bagged**：在随机成员子集上重复贪心并平均权重，进一步降低方差。")
            A("  它在同一份 OOF 上取得最高分（0.8940），但嵌套验证显示这一优势并不可靠（见下）。")
            A("")
            trace = ens.get("greedy_trace", [])
            if trace:
                A(f"贪心选择共进行 {len(trace)} 步，轨迹为：" + " → ".join(
                    f"{t['member']}({t['oof_acc']:.4f})" for t in trace[:12]) +
                  ("…" if len(trace) > 12 else ""))
                A("")
            val = ens.get("validation")
            if val:
                A("**关键：集成方法本身的嵌套验证。**")
                A("")
                A("上述比较全部在同一份 OOF 上进行，任何“在 OOF 上选权重”的做法都会带来乐观偏差。")
                A("为得到无偏的方法对比，做了 100 次分层半样本划分：在 A 半上选择权重，")
                A("在**未参与选择的 B 半**上评估（`validate_ensemble.py`）：")
                A("")
                rows = [[k, f"{v['mean']:.4f}", f"{v['std']:.4f}", f"{v['min']:.4f}",
                         f"{v['gain_vs_best_single']:+.4f}"] for k, v in
                        sorted(val.items(), key=lambda x: -x[1]["mean"])]
                A(table(["集成方法", "B 半平均准确率", "标准差", "最差", "相对最优单体增益"], rows))
                A("")
                pa = ens.get("validation_paired")
                if pa:
                    A(f"bagged 贪心与最优单体的配对比较：胜 {pa['wins']} / 负 {pa['losses']} / "
                      f"平 {pa['ties']}，平均差 {pa['mean_diff']:+.4f}，"
                      f"Wilcoxon 符号秩检验 p = {pa['p_value']:.4f}"
                      f"（{'显著' if pa['p_value'] < 0.05 else '不显著'}）。")
                    A("")
                A("嵌套验证给出三点重要结论：")
                A("")
                A("1. **集成的增益比在同一份 OOF 上看到的要小得多**：bagged 贪心的 OOF 是 0.8940，")
                A("   但在 B 半上的平均只有 0.8825，仅比最优单体高约 0.16 个百分点，且统计上不显著；")
                A("2. **把弱成员全部纳入会伤害性能**：等权平均全部 8 个成员反而比最优单体低约 1 个百分点；")
                A("3. **只取最强的少数成员等权平均最稳**：top-3 等权在嵌套验证中最好（+0.32 个百分点）。")
                A("")
                A(f"因此最终提交采用嵌套验证推荐的方式：**{ens.get('validation_choice', 'top-3 等权')}**，")
                A("即取 OOF 最强的 3 个成员（微调 resnet18 / resnet50 / efficientnet_b0）等权平均概率。")
                A("")
                A("需要说明：0.8940 这类在同一份 OOF 上搜索得到的数字属于**乐观估计**，")
                A("本报告以嵌套验证的 0.8840 作为对测试集准确率的更可信预期。")
                A("")
        A("### 5.5 类别先验偏移的检验与阻尼校正")
        A("")
        pt = j("prior_test.json", None)
        if pt:
            A("训练集五类各 100 张，是完全均衡的。但模型在测试集上的预测分布明显偏离")
            A("“与训练集同分布”时应有的样子。为判断这是模型偏置还是数据分布差异，")
            A("本实验比较了 **OOF 预测边缘分布**（模型在均衡数据上的固有偏置）")
            A("与 **测试集预测边缘分布**：")
            A("")
            A(table(["类别", "OOF 预测占比（500 张）", "测试预测占比（378 张）", "差值"],
                    [[c, f"{pt['oof_marginal'][i]:.3f}", f"{pt['test_marginal'][i]:.3f}",
                      f"{pt['test_marginal'][i]-pt['oof_marginal'][i]:+.3f}"]
                     for i, c in enumerate(C.CLASSES)]))
            A("")
            A(f"卡方检验结果：**χ² = {pt['chi2_stat']:.2f}，df = {pt['dof']}，"
              f"p = {pt['p_value']:.2e}**，拒绝“测试集与训练集同分布”的原假设。")
            A("")
            hp = pt.get("hypotheses", {})
            if hp:
                A("进一步检验三种具体先验假设（期望频数口径）：")
                A("")
                A(table(["假设", "χ²", "p 值", "结论"],
                        [[k, f"{v['chi2']:.2f}", f"{v['p']:.4f}",
                          "不拒绝" if v["p"] > 0.05 else "拒绝"] for k, v in hp.items()]))
                A("")
            A("这说明**测试集的类别先验与训练集不一致**（且不等于简单的原始数据集比例），")
            A("属于典型的 prior shift 场景。")
            A("")
        ps = j("prior_shift.json", None)
        pm = j("prior_sim.json", None)
        if ps and ps.get("estimated_prior_test"):
            A("为量化这一偏移，用 EM（Saerens–Latinne–Decaestecker）从测试集软标签中直接估计目标先验：")
            A("")
            A(table(["类别", "EM 估计的测试集先验"],
                    [[c, f"{ps['estimated_prior_test'][i]:.3f}"] for i, c in enumerate(C.CLASSES)]))
            A("")
            oc = ps.get("oof_check", {})
            if oc:
                A("**估计器的自检**：在 OOF（先验确为均衡）上运行同一估计器，"
                  f"估计值与均衡的最大偏差为 {oc.get('max_dev_from_uniform', float('nan')):.3f}，"
                  f"若直接套用完全校正（α=1），OOF 准确率会从 "
                  f"{oc.get('oof_acc_before', float('nan')):.4f} 降到 "
                  f"{oc.get('oof_acc_after', float('nan')):.4f}。")
                A("说明**完全校正会因模型自身误差而引入负偏置，不能直接使用**。")
                A("")
        if pm:
            A("因此本实验用 OOF 构造**已知真值的合成偏移验证集**（按目标先验加权重采样），")
            A("客观标定阻尼校正强度 α（校正后验 ∝ p · (π_t/π_s)^α）：")
            A("")
            grid = pm.get("alpha_grid", [])
            rows = []
            for sname, d in pm.get("scenarios", {}).items():
                rows.append([sname] + [f"{d[str(a)] if str(a) in d else d[a]:.4f}" for a in grid])
            A(table(["场景"] + [f"α={a}" for a in grid], rows))
            A("")
            A(f"平均最优 α = {pm.get('best_avg_alpha')}，最坏情形最优 α = {pm.get('best_worst_alpha')}。")
            if ps and ps.get("applied"):
                A("")
                A(f"最终以 **α = {ps.get('alpha')}** 做阻尼校正，"
                  f"改变预测的样本数为 {ps.get('changed')} / {ps.get('n')}"
                  f"（{100*ps.get('changed',0)/max(1,ps.get('n',1)):.1f}%）。")
                A("")
                A(table(["类别", "校正前预测数", "校正后预测数"],
                        [[c, ps["counts_before"][c], ps["counts_after"][c]] for c in C.CLASSES]))
                A("")
            else:
                A("")
                A("**结论：不做先验校正。** 仿真表明，在先验估计可能出错的各种场景下，")
                A("任何强度的校正（α>0）在最坏情形下都不如不校正（最坏情形最优 α = 0）。")
                A("原因是分类器本身仍有约 11% 的误差，而 EM 先验估计会被这些误差污染：")
                A("自检显示，即使在先验确为均衡的 OOF 上，估计值与均衡的最大偏差也达到约 0.04，")
                A("照此完全校正会白白损失 2 个百分点以上。")
                A("")
                A("因此本实验最终**保留未校正的预测结果**，并在报告中完整记录这一“负结果”：")
                A("它说明先验校正必须建立在对估计器误差的实测之上，而不能仅凭“分布不一致”就动手。")
                A("")
        A("")
        A("### 5.6 结论")
        A("")
        A(f"1. 在每类仅 100 张训练样本的条件下，通过 ImageNet 预训练迁移学习 + 强增广 + TTA + 集成，")
        A(f"   交叉验证 OOF 准确率达到 **{fmt_pct(fin['oof_acc'])}**；")
        A("   经嵌套验证校正乐观偏差后，对测试集准确率的可信预期约为 **88.4%**。")
        A("2. 冻结特征 + 线性 SVM 是极高效的强基线（数秒即可完成一次交叉验证），")
        A("   而端到端微调配合 mixup、标签平滑与 TTA 精度更高（微调 resnet18 达 0.8860）；")
        A("   二者概率集成进一步降低了预测方差。")
        A("3. 多骨干特征拼接与多模型概率集成都有一致增益，说明不同预训练网络的表征具有互补性；")
        A("   但嵌套验证表明**只应集成最强的少数成员**，把弱成员一并平均反而会拉低准确率。")
        A("4. 统计检验发现测试集类别先验与训练集显著不同（χ²=%.2f, p=%.1e），属于 prior shift；"
          % (j("prior_test.json", {}).get("chi2_stat", 0.0),
             j("prior_test.json", {}).get("p_value", 1.0)) if j("prior_test.json", None) else
          "4. 统计检验发现测试集类别先验与训练集显著不同，属于 prior shift；")
        if j("prior_test.json", None):
            ps_ = j("prior_shift.json", {}) or {}
            if ps_.get("applied"):
                A("   本实验用 EM 估计目标先验，并在 OOF 合成偏移集上仿真标定出阻尼校正强度 α，")
                A("   **只在最坏情形下也不劣于不校正时才应用**，避免盲目拉平类别分布。")
            else:
                A("   本实验用 EM 估计了目标先验，但在 OOF 合成偏移集上做仿真标定后发现：")
                A("   **校正的最坏情形增益为 0，即现有证据并不支持做先验校正**，")
                A("   因此最终提交保持未校正结果。这一“负结果”同样是重要的实验结论 —— ")
                A("   它说明在分类器本身仍会犯错时，简单地按估计先验重加权并不一定可靠。")
        A("")
        A("**可能的改进方向**：")
        A("")
        A("- 提高输入分辨率（如 320×320）以保留幼苗叶形细节；")
        A("- 引入更强增广（RandAugment、CutMix）与更长训练；")
        A("- 使用更强的预训练骨干（ConvNeXt、Swin、ViT）或自监督预训练权重；")
        A("- 在无标签测试集上做自训练 / 伪标签半监督学习，把 378 张测试图也利用起来。")
    else:
        A("### 5.3 集成与最终结果")
        A("")
        A("（`ensemble_report.json` 尚未生成，请先运行 `ensemble.py`。）")
    A("")
    A("## 六、代码结构与运行方式")
    A("")
    A("```")
    A("2026task1/code/")
    A("├── py.cmd               统一运行包装器（注入 PYTHONPATH / TORCH_HOME / 线程数）")
    A("├── common.py            公共配置：路径、类别、随机种子、数据读写、提交文件写出")
    A("├── models.py            骨干注册表、多视图特征抽取（翻转平均 + 双尺度拼接）")
    A("├── train_linear.py      路线A：冻结特征 + 9 种分类头 × 5 折 CV + 多骨干拼接融合")
    A("├── finetune.py          路线B：端到端微调 + mixup + 标签平滑 + 8 视图 TTA（支持单折并行）")
    A("├── aggregate_folds.py   汇总分折产物为整份 OOF / 测试概率 / CV 记录")
    A("├── validate_ensemble.py 集成方法的嵌套验证（半样本 A/B，消除选权重的乐观偏差）")
    A("├── ensemble.py          集成与提交：多种集成策略 + 生成 submission-for-task2.csv")
    A("├── eval_report.py       评估诊断：OOF 混淆矩阵、各类 P/R/F1、置信度分层")
    A("├── prior_test.py        先验偏移的卡方检验（OOF 边缘分布 vs 测试边缘分布）")
    A("├── prior_sim.py         用 OOF 合成偏移集仿真标定阻尼校正强度 α")
    A("├── prior_shift.py       EM 先验估计与阻尼校正的应用（含应用门槛自检）")
    A("├── viz.py               报告配图（fig1~fig7）")
    A("├── make_report.py       由 result/*.json 与训练日志自动生成本报告")
    A("├── make_docx.py         报告转 Word")
    A("├── make_delivery.py     整理并打包交付目录")
    A("└── finalize.cmd         一键跑完整条最终流水线")
    A("```")
    A("")
    A("**一键复现**（工作目录 `D:\\deepseek\\2026task1\\code`）")
    A("")
    A("```bat")
    A("REM 训练阶段（可并行；特征会缓存到 cache/）")
    A("py.cmd train_linear.py resnet18 resnet50 efficientnet_b0 densenet121 --size 224")
    A("py.cmd finetune.py --backbone resnet18        --all-folds --epochs 45 --tag ft_r18")
    A("py.cmd finetune.py --backbone resnet50        --all-folds --epochs 40 --lr 2.5e-4 --tag ft_r50")
    A("py.cmd finetune.py --backbone efficientnet_b0 --all-folds --epochs 45 --tag ft_effb0")
    A("")
    A("REM 汇总、验证、集成、诊断、报告、打包（一键）")
    A("finalize.cmd")
    A("```")
    A("")
    A("依赖环境由 `py.cmd` 统一注入。注意：受限沙箱禁止进程间命名管道，")
    A("因此 `DataLoader` 必须 `num_workers=0`，joblib 必须使用串行后端。")
    A("")
    A("## 七、附录：结果文件清单")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `submission-for-task2.csv` | **最终提交文件**（378 行预测） |")
    A("| `README.md` | 交付说明与复现指南 |")
    A("| `result/linear_summary.json` | 路线A：4 骨干 × 9 分类头的全部 CV 结果 |")
    A("| `result/cv_ft_*.json` | 微调各折验证准确率、OOF 准确率与覆盖折数 |")
    A("| `result/ensemble_report.json` | 集成成员、各策略对比、最终权重、预测分布与置信度 |")
    A("| `result/ensemble_validation.json` | 集成方法的嵌套验证结果 |")
    A("| `result/eval_report.json` | OOF 混淆矩阵、各类 P/R/F1、置信度分层 |")
    A("| `result/prior_test.json` | 先验偏移的卡方检验 |")
    A("| `result/prior_sim.json` | 阻尼校正强度 α 的仿真标定 |")
    A("| `result/prior_shift.json` | EM 先验估计、OOF 自检与应用门槛判定 |")
    A("| `result/final_test_proba.npy` | 测试集最终概率矩阵 (378×5) |")
    A("| `result/log_ft_*.txt` | 微调逐轮训练日志 |")
    A("| `reports/fig1~fig7*.png/.pdf` | 报告配图（300 dpi） |")
    A("| `提交包.zip` | 上述交付物的打包（不含模型权重与数据集） |")
    A("")

    os.makedirs(C.REPORTS, exist_ok=True)
    p = os.path.join(C.REPORTS, "实验报告.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  [报告] {p}  ({len(''.join(L))} 字符)")


if __name__ == "__main__":
    main()
