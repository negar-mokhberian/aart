# 🔍 Welcome to AART: Annotator-Aware Representations for Texts

This repository contains the code for **Annotator-Aware Representations for Texts (AART)**, introduced in the paper:  
[📄 Capturing Perspectives of Crowdsourced Annotators in Subjective Learning Tasks](https://aclanthology.org/2024.naacl-long.407/#), accepted at **NAACL 2024** (main track).  

AART is designed to improve subjective classification tasks by incorporating annotator-aware representations, addressing biases in label aggregation, and enhancing fairness in model performance across diverse annotators.

---

## 📌 Abstract
Supervised classification heavily depends on datasets annotated by humans. However, in subjective tasks such as toxicity classification, these annotations often exhibit low agreement among raters. Annotations have commonly been aggregated by employing methods like majority voting to determine a single ground truth label. In subjective tasks, aggregating labels will result in biased labeling and, consequently, biased models that can overlook minority opinions. Previous studies have shed light on the pitfalls of label aggregation and have introduced a handful of practical approaches to tackle this issue. Recently proposed multi-annotator models, which predict labels individually per annotator, are vulnerable to under-determination for annotators with few samples. This problem is exacerbated in crowdsourced datasets. In this work, we propose Annotator Aware Representations for Texts (AART) for subjective classification tasks. Our approach involves learning representations of annotators, allowing for exploration of annotation behaviors. We show the improvement of our method on metrics that assess the performance on capturing individual annotators’ perspectives. Additionally, we demonstrate fairness metrics to evaluate our model’s equability of performance for marginalized annotators compared to others.

---

## 🚀 Getting Started

### 🔧 Installation

Ensure you have Python installed. Then, clone this repository and install the required dependencies:

```bash
git clone https://github.com/negar-mokhberian/aart.git
cd aart
pip install -r requirements.txt
```

---

## 🏗️ How to Use

### 1️⃣ Running AART

You can explore available command-line arguments using:

```bash
python main.py --help
```

A sample execution is provided in [`test_run.sh`](./test_run.sh). Modify it according to your dataset and approach.

**Example Run Command:**
```bash
python main.py --data_name my_dataset --approach aart
```

- `--data_name`: A custom name for your dataset.
- `--approach`: Choose from `"single"`, `"multi_task"`, or `"aart"`.

---

## 📂 Dataset Format

The dataset should be stored under:

```
./data/APPROACH/DATA_NAME/all_data.csv
```

where:
`APPROACH` and `DATA_NAME` are both provided as input arguments. 
- `APPROACH` corresponds to the selected method (`single`, `multi_task`, or `aart`).
- `DATA_NAME` is a user-defined dataset name.

### Expected Columns

| Column Name   | Description |
|--------------|------------|
| **prep_text**  | Preprocessed texts. Apply your preferred preprocessing method before storing |
| **text_id**    | A unique numerical ID for each text instance |
| **annotator**  | A unique identifier for each annotator (e.g., `annotator_0`, `annotator_1`, ...) |
| **label**      | The annotation provided by the respective annotator for the given text |

---

## 📜 Citation

If you use AART in your research, please cite:

```bibtex
@inproceedings{mokhberian-etal-2024-capturing,
    title = "Capturing Perspectives of Crowdsourced Annotators in Subjective Learning Tasks",
    author = "Mokhberian, Negar  and
      Marmarelis, Myrl  and
      Hopp, Frederic  and
      Basile, Valerio  and
      Morstatter, Fred  and
      Lerman, Kristina",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.407/",
    doi = "10.18653/v1/2024.naacl-long.407",
    pages = "7337--7349",
}

```
⭐ Also, if you use this repository, please consider giving us a star on GitHub! ⭐
---
