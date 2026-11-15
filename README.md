# Disease Entity Normalization in Chinese Electronic Medical Records

Code accompanying **Disease Entity Normalization in Chinese Electronic Medical Records: A Knowledge Graph Query Utility Evaluation**.

The workflow combines RoBERTa–BiLSTM–CRF entity recognition, ICD-based disease-name normalization, record-level co-occurrence graph construction, and paired disease-to-drug and disease-to-procedure query evaluation. Neo4j provides interactive graph exploration.

## Setup

```bash
git clone https://github.com/Yoliiilm/emr-kg-normalization.git
cd emr-kg-normalization
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before running the scripts. CUDA and a compatible PyTorch build are needed for GPU training. Individual experiments may require additional model-specific dependencies.

## Data and models

Obtain CCKS2019 Yidu-S4K, CCKS2017 clinical NER, CHIP-CDN, and the ICD-10 Beijing Clinical Edition v601 terminology through their respective providers and terms. Place the resources in the paths expected by the scripts. The source code documents the expected directory structure and command-line arguments.

This repository contains source code and the experiment protocol. Clinical records, model checkpoints, third-party terminology tables, credentials, and local experiment outputs are not distributed here. Re-running trained-model experiments requires the corresponding data, checkpoints, and configuration; a source-only checkout does not reproduce the reported measurements by itself.

## Main commands

```bash
python src/eval_g_raw_vs_g_norm.py --help
python src/eval_query_level.py --help
python src/eval_calibrator.py --help
python src/eval_cross_dataset.py --help
python src/eval_ablation.py --help
```

Read `docs/EXPERIMENT_PROTOCOL.md` for evaluation definitions, data-use history, frozen configurations, and statistical procedures. Some scripts read fixed local paths; inspect their configuration before execution.

For local Neo4j imports, set `NEO4J_PASSWORD`, and optionally `NEO4J_URI` and `NEO4J_USER`. Keep credentials outside the repository. Read the import arguments before using operations that replace existing graphs.

## Evaluation scope

The main comparison holds predicted entities, co-occurrence rules, and query code constant. Query F1 measures agreement with an operational reference obtained by applying the study's normalization to manually annotated entities. CHIP-CDN strict name matching and cross-dataset analyses are reported separately. The protocol records earlier development-set inspection and retrospective rule development.

## Author

Wang Youlimei, Computer Information Systems, Wenzhou-Kean University.

## 中文说明

本仓库为中文电子病历疾病实体标准化与知识图谱查询效用论文的配套源码。代码涵盖实体识别、疾病名称归并、共现建图、Neo4j导入及评价。数据、模型和第三方词表需依据原提供方条件自行取得；具体评价口径与实验历史见实验协议。
