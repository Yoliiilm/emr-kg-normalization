# 实验协议与可复核性审计

更新时间：2026-10-25

本文档规定论文中可作为正式证据的实验口径。所有未达到该口径的历史结果仍保留在 `outputs/metrics/`，但必须标为诊断性结果，不能与正式主结果混写。

## 1. 数据与样本量

| 任务 | 数据与划分 | 正式评估样本 |
|---|---|---:|
| NER | CCKS2019：900 train / 100 dev / 379 test 病历 | test；NER micro-F1 |
| 图谱查询 | CCKS2019 test 预测图 | 使用药物 482 个配对查询 / 273 个疾病簇；实施手术 387 个配对查询 / 195 个疾病簇 |
| CHIP-CDN 标准化 | 6000 train / 2000 dev；test 10000 条标签为空 | CHIP-CDN dev 为正式标准答案基准；策略比较只用 train 内 OOF；test 不报告准确率 |
| CCKS2017 迁移 | 300 份病历、236 个不同疾病写法、722 次疾病提及 | 查询和迁移诊断；没有标准名金标，不报告跨院标准化准确率 |
| 跨医院独立标准名金标 | 当前没有可用的逐条“原写法—标准名/概念 ID”标注集 | 不报告跨医院标准化准确率；不把待标注材料或自动映射当作金标 |
| 链接预测 | CCKS2019 train 的 900 份病历；病历级 5 折重复 5 次 | 每次 548--608 条配对未见边；严格共同候选集每次 435--483 条 |

### CHIP-CDN dev 的正式角色

CHIP-CDN dev（2000 条，含人工标准名/概念标签）是本研究正式标准答案基准：报告严格标准名匹配、覆盖率、接受精确率、候选召回及来源分层。它支持的是与 CHIP-CDN 标注口径的一致性，不等于跨医院独立验证。训练和阈值选择只用 CHIP-CDN train 内部切分/外层折。dev 已在此前开发过程中被查看过，因此从本次冻结日起作为固定留出终评集停止用于任何规则、阈值、来源策略或模型选择；它不是历史上从未接触的盲测集。官方 `CHIP-CDN_test.json` 的 10000 条标签为空，不报告其准确率。

正式评估不得启用 `data/dicts/chip_cdn_alias.json`：该缓存含 train/dev 派生别名。应使用 `src/eval_calibrator.py` 默认设置，它会关闭全量 CHIP-CDN 别名并只安装 train 的 fit 子集别名（结果字段 `alias_source=train_fit_only`）。

## 2. 数据泄漏控制

- 标准化校准：候选生成、别名安装、模型拟合、阈值选择均只使用训练折；外层折只做评估。
- `min_source_positives=20`、`min_source_negatives=50` 固定，不因 dev 结果降低。
- CHIP-CDN dev 只做最终一次评估，不参与正式阈值、来源策略或收缩强度选择。
- 校准器在 CCKS2019 test 上只做最终评估；但历史完整规则和人工别名曾参考 CCKS2019 test 的错误分析，因此该主查询 F1 是历史描述性主结果，不能表述为完全未见测试集终评。
- 无测试表面信息的外部查询复验只在 train+dev 上选配置，并在 `outputs/metrics/experiments/statistical_reproducibility/` 单独报告 test 结果；该复验与历史完整规则图查询分开引用。
- 链接预测每折只用训练病历建立映射、图和模型；raw/norm 使用同一批配对留出边。
- 本项目没有完成 CCKS2017 逐条标准名/概念 ID 的独立标注；不得把准备过的抽样表或封存的系统预测描述成已完成的独立金标评估。

## 3. 随机性与重复次数

| 实验 | 随机/重复设置 | 解释 |
|---|---|---|
| NER 主模型 | seed=42 | 单一训练种子；结果应报告这一限制，不冒充多种子稳健性 |
| SapBERT 训练 | seed=42 | 候选检索器单种子；校准策略用 5 折 train-only OOF 复核 |
| 图查询差异 | bootstrap 2000 次；置换 10000 次；符号翻转检验 | 以查询或疾病簇为重采样单位 |
| 校准策略 | 5 折嵌套 train-only；固定收缩强度网格 `0,10,25,50,100,250,1000` | 外层结果用于比较，不用 dev 选策略 |
| 链接预测 | seed 42--46 的重复留出；另有病历级 5 折 × 5 次 | 仅把后者作为主要稳定性分析；t 区间是重复划分的描述性区间 |
| 多重比较 | 38 个统一整理的比较家族（非预注册）；Holm 与 BH | 主结论优先报告 Holm；BH 结果单独标注，并说明比较家族是事后整理 |

**仍存在的稳健性缺口：** NER 主模型和 SapBERT 本身没有多训练种子重跑；这属于需要在论文局限中明确的剩余风险，而不是用现有单种子结果推断训练稳定性。

## 4. 置信区间与显著性

- 查询 F1：按查询和按疾病簇分别 bootstrap 2000 次，报告 95% CI；另做 10000 次随机置换和双侧符号翻转检验。
- 校准策略：按病名表面聚类的配对 bootstrap 2000 次，报告覆盖率和严格接受精确率差值及 95% CI。
- 外部支持：按疾病簇 bootstrap，报告支持率差值及 95% CI；外部库支持不等于医学正确率。
- 链接预测：报告每次重复的逐次差值、均值和描述性 95% t 区间；由于重复折并非独立样本，不将其称为独立临床显著性检验。
- 多重比较：正式论文以 Holm 校正后的结论为主，BH 仅作为补充。

历史完整流程的核心查询结果：用药 F1 `0.511 -> 0.839`，差值 95% CI `[+0.263,+0.379]`；手术 F1 `0.378 -> 0.653`，差值 95% CI `[+0.197,+0.343]`。两项在现有比较家族的 Holm 校正后仍显著，但参考图采用本项目归一化、完整规则曾参考测试集错误分析，不能单独证明标准名正确或替代无泄漏终评。

## 5. 正式基线清单

### 标准化与校准

1. 不标准化（raw）；
2. 仅规则；
3. ICD 词典/规则/人工别名；
4. 本方法完整流程；
5. RoBERTa 检索；
6. SapBERT 检索；
7. CHIP-CDN 监督模型；
8. Jaccard、Jaccard+BERT；
9. 大模型候选选择；
10. global、source-specific、shrinkage；
11. `chip` 严格拒识；
12. raw score、词面相似度、来源先验阈值。

### 链接预测

1. TransE；
2. DistMult；
3. 训练图药物度数；
4. 共同邻居；
5. 疾病邻居传播；
6. 疾病集合 Jaccard、余弦、资源分配。

链接预测的边来自病历共现，不是临床治疗金标；因此只能作为图结构补全分析。

## 6. 失败案例与不确定结果

- CCKS2019 664 个疾病表面词中，233 个原样保留；主要原因是粒度过细、检查所见、无部位组织学词和复合诊断。
- 归一化后查询 F1 下降的查询很少，但存在心功能分级、术后状态、左右侧和肿瘤亚型被合并到上位病名的案例；完整清单见 `outputs/metrics/error_analysis.txt`。
- CHIP-CDN dev 2000 条中，错误链接 69 条，未链接 1143 条；这证明本方法的主要代价是覆盖率，而非可忽略的误合并。
- `chip` 来源负例不足 50，不能拟合可靠的来源专属模型；五折结果支持严格拒识，而不支持把它作为收缩模型的独立贡献。
- 校准 5 折中没有达到预设 85% 精确率目标；SapBERT 主要提高候选召回，尚未证明层级收缩本身带来稳定增益。
- 链接预测中简单度数/邻居基线强于部分嵌入模型；TransE 的共同候选集收益接近 0，不能写成普遍链接预测提升。

## 7. 阈值与结果分层

校准方案的正式主线只引用 `train-only` 或 `nested train-only` 的阈值和策略选择，以及最终一次 dev 终评。对于图查询，必须区分历史完整流程结果与无测试表面信息的外部查询复验。`src/eval_supervised_threshold.py` 及其 `supervised_threshold.*` 是早期诊断实验：脚本在 CHIP-CDN dev 网格选阈值，不能作为无泄漏主结果；它们保留用于追溯和失败分析。

正式主结果入口：

- `outputs/metrics/bootstrap_query_f1.*`
- `outputs/metrics/experiments/source_shrinkage/`
- `outputs/metrics/experiments/algorithm_contribution/`
- `outputs/metrics/experiments/link_prediction/repeated_record_cv/expanded_cv.*`
- `outputs/metrics/experiments/statistical_reproducibility/`

## 8. 尚未完成项

1. CHIP-CDN dev 是正式标准答案基准；但跨医院独立标准名金标目前不可用，因此不能报告 CCKS2017 跨医院标准化准确率。没有标注结果时，不声称标注材料已封存或已完成盲评。
2. NER 和 SapBERT 的多训练种子复验尚未完成。
3. 历史完整规则曾参考 CCKS2019 test 错误，且查询参考图仍部分依赖本项目归一化；应将其作为评价边界写入论文，而不是隐藏。
4. 链接预测和层级收缩只能作为探索性结果，不能作为已证实的核心收益。

## 9. 系统冻结与外部数据集审查

- 冻结日期：2026-10-25。核心规则、词典、检索器、SapBERT 校准模型、校准训练/评估脚本及 CHIP-CDN/CCKS 输入文件以 SHA-256 清单固定，见 `outputs/metrics/experiments/system_freeze/frozen_system_manifest.json`。
- 验证冻结状态：`.venv/Scripts/python.exe src/freeze_experiment.py --verify`。创建快照时脚本拒绝覆盖既有 manifest。
- NCBI Disease Corpus 的官方 Public Domain Notice 明确数据可供公众自由使用和复制。2026-10-25 下载了 train/dev/test 三个官方 split，压缩包及 SHA-256 见 `outputs/external_data/ncbi_disease_corpus/download_manifest.json`。冻结模型与词表按中文/ICD 构建，不能直接映射该英文 MeSH/OMIM 金标；因此没有把其分数冒充中文系统外部准确率，而是另做隔离的英文词面基线。来源：<https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/>。
- NCBI 英文词面实验由 `src/experiment_ncbi_external.py` 运行：训练别名只来自 train，字符 TF-IDF 阈值只按 dev 的 `SpecificDisease` 子集上接受精确率至少 85% 时的最大覆盖选择，test 不参与选择。计分单位为带概念 ID 的标注提及；置信区间按 PMID 文献聚类 bootstrap 2000 次。数据量：train/dev/test 分别为 5,145/787/960 个提及（592/100/100 篇摘要）；test 的 555 个 `SpecificDisease` 上，train 精确别名覆盖率 57.30%、接受精确率 97.48%；dev 定阈值的字符检索覆盖率 69.19%、接受精确率 85.94%。全部四类标注合并的 test 覆盖率 72.60%、接受精确率 87.52%，文献级 bootstrap 95% CI 分别为覆盖率 [65.00%, 79.42%]、接受精确率 [83.89%, 90.97%]。机器可读结果为 `outputs/metrics/experiments/external_validation/ncbi_lexical_baseline.json`。该实验只证明简单英文词面基线在 NCBI 上的外部可行性，不验证冻结中文系统，也不是本文标准化方法的跨语言效果证明。
- BC5CDR 也已取得并完成独立英文基线。原始压缩包中的 NCBI Public Domain Notice 明确允许公众使用和复制；来源、SHA-256 见 `outputs/external_data/bc5cdr/download_manifest.json`。`src/experiment_bc5cdr.py` 只保留带 MeSH ID 的 Disease 提及，排除 Chemical 与 ID `-1`；train 建别名，dev 的疾病提及选 85% 目标精确率下的最大覆盖阈值，test 只终评。train/dev/test 分别有 4,057/4,141/4,310 条疾病提及（499/498/500 篇文献）。test 精确查表覆盖率 65.38%、接受精确率 98.01%；字符 TF-IDF 阈值检索覆盖率 81.79%、接受精确率 85.76%，总正确数从 2,762 增至 3,023。文献级 bootstrap 2000 次的阈值结果 95% CI：接受精确率 [83.86%, 87.78%]、覆盖率 [79.70%, 83.82%]、端到端准确率 [67.72%, 72.62%]。完整结果见 `outputs/metrics/experiments/external_validation/bc5cdr_lexical_baseline.json`。与 NCBI 一样，它是英语 MeSH 语料的外部词面基线，不是冻结中文 ICD 系统的评估。
- MedMentions 已按用户授权仅作本地实验，未分发原文。BigBio 数据卡标记 CC0-1.0，但其原始 GitHub 仓库没有明确许可证；这一冲突已记录在 `outputs/external_data/medmentions/download_manifest.json`，任何公开发布前须先解决许可状态。语料使用官方 train/dev/test PMID 清单；仅评估 T047（Disease or Syndrome）和 UMLS CUI。train/dev/test 分别为 7,099/2,408/2,243 条提及（2,635/878/879 篇文献），训练别名 2,118 个；dev 按接受精确率目标 85% 选择最大覆盖阈值 0.9031（1,012 条接受，精确率 85.67%），test 不参与选择。test 精确词面查表覆盖率 55.55%、接受精确率 94.46%；字符 TF-IDF 阈值检索覆盖率 40.79%、接受精确率 91.69%，文献级 bootstrap 2000 次 95% CI 分别为覆盖率 [36.42%, 44.88%]、接受精确率 [87.72%, 95.03%]、端到端准确率 [32.94%, 41.71%]。机器可读结果为 `outputs/metrics/experiments/external_validation/medmentions_t047_lexical_baseline.json`。它是英语 UMLS-CUI 词面基线，不评估冻结中文 ICD 系统。来源：<https://github.com/chanzuckerberg/MedMentions>、<https://huggingface.co/datasets/bigbio/medmentions>。
- CCL2026-Eval 的官方页面文本显示 2600 份脱敏病历，train/A/B 分别 1800/400/400 条，只有 train 与 A（验证集）公开，B（测试集）不公开；报名后还需邮件提交数据申请承诺书。A/B/C ZIP 约 549 KB / 1 MB / 685 KB。页面字段含主要/其他诊断编码及手术编码，但标签是记录级 ICD 编码，不含逐条 mention-to-concept 对齐。由于赛事报名已截止，且数据申请仍须承诺书，本项目目前无法取得该数据；可作为条件许可下的外部 ICD 编码效用候选，不是当前可运行的独立病名标准化金标。使用前仍需核验承诺书用途、发表/再发布限制、真实字段和切分。
- JointCoder 的公开仓库声称其中文 EHR 编码数据含 6,747 条住院记录及 ICD-10/ICD-9-CM-3 标注，但仓库文件清单中只有 6 条示例记录，没有完整数据集；仓库没有机器可读的数据许可。故不能视为可公开取得的外部验证集。来源：<https://github.com/KJOrigin/JointCoder>。
- BearLiX/CMCN 的仓库公开了五折 CSV/TXT；经用户明确限定为本地结构检查后，仅下载 fold-1 的 `DNN.test.csv` 到 `outputs/external_data/cmcn_unverified/`。该文件 4,289 行、381 个自由文本标签、`label/content` 两列及三个空表头；检查样本中没有识别为 ICD、MeSH、UMLS CUI 或 OMIM 的标签编号。仓库未说明原始数据来源、标签是否为标准病名、概念映射或再利用许可；因此该文件没有用于训练、计分或发表，正文也未复制。检查范围和 SHA-256 见同目录 `inspection_scope.json`。Yifan-haddock/CMCN 仓库虽为 MIT 许可，但公开的是代码而非这套五折数据。来源：<https://github.com/BearLiX/CMCN>、<https://github.com/Yifan-haddock/CMCN>。
- 综合排查，截至 2026-10-25，没有找到同时满足“中文疾病 mention—标准概念 ID 逐条金标、来源独立、公开许可清楚、可直接取得、未参与训练/调参”的新数据集。JointCoder/CCL2026 可作为取得授权后的记录级 ICD 编码下游研究候选，但不可替代 mention-level 标准化金标。不得用自动映射、来源不明的数据或跨语言结果宣称已完成冻结中文系统的独立标准化验证。

### CHIP 手术名称辅助评测（2026-10-26）

- 本地已有 `data/dicts/raw/手术标注词2500.tsv`，每行包含手术写法、ICD-9 编码和标准术式名，共 2,500 条写法、1,089 个编码/标准名。本轮不重新下载数据，不改系统冻结文件或主结果；输入 SHA-256、评测协议、完整结果见 `outputs/metrics/experiments/surgery_ben/complete_eval_20261026.json` 和同名 `.txt`，复现命令为 `.venv/Scripts/python.exe src/experiment_surgery_ben.py`。
- 冻结归一化器在全量上的严格 ICD-9 命中为 88/2,500（3.52%），与原样保留基线一致；自动接受（`surgery_alias`/`surgery_norm`）覆盖 258/2,500（10.32%），其中严格编码命中 1/258（0.39%）。按 ICD-9 概念簇配对 bootstrap 5,000 次，冻结归一化相对原样基线的准确率差为 0，95% CI [0, 0]。当前术式规则没有显示跨资源迁移收益。
- 固定字符 TF-IDF（字符 2–4 gram；未在该表调参）以表内 1,089 个标准术式名作为闭集候选词表，Hits@1/3/5 为 44.92%/63.52%/70.04%。候选清单来自同一数据资源的标准名列，因此该结果仅作闭集检索诊断，不能宣称是独立候选本体上的外部泛化，也不能与冻结归一化器结果混作同一方法成绩。
- 当前 `surgery_alias.json` 声明别名为 CCKS2019 来源，评测写法与别名键的精确重叠为 0；这只能排除字面重叠，不能排除规则层面的同源影响。该 TSV 原始许可/再利用条款尚未核实，发表或再分发前须确认许可。此实验只评估手术 mention→ICD-9，不是疾病名称标准化准确率，也不是给定整份病历预测 ICD 编码。
- CCL2026 中文病历 ICD 编码数据仍未取得。官方任务仓库规定签署数据使用与保密承诺书并邮件申请，赛事报名已截止；当前只能联系组织方询问是否接受赛后科研申请。未经授权不得把仓库样例或网页示例当评测数据。
