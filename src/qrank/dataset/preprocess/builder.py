from __future__ import annotations
import json
from pathlib import Path
from typing import List, Literal
import os
import pandas as pd
from datasets import Dataset, DatasetDict
import numpy as np
from .sql.builder import SQLDatasetBuilder
from dotenv import load_dotenv
load_dotenv()

ABBREVIATIONS = {
    "schema": "scm",
    "sql_query": "sql",
    "question": "q",
    "embedding": "emb",
    "answer_feature": "af",
    "agreement": "agr",
    "correlation": "corr",
    "ranking": "rk",
    "label": "lbl",
    "remove_syntax_errors": "nse",
    "set_semantic": "ss",
    "classes": {2: "c2", 3: "c3", 4: "c4"},
    "Arctic-R1-7B": "A", 
    "XiYanSQL-7B": "X", 
    "OmniSQL-7B": "O",
    "QWen2.5-7B": "Q",
    "rnj-1-8B": "R",
    "SQLCoder-7B": "S",
    "natural-sql-6.7B": "N"
}


def compute_dataset_metrics(dataset):
    labels_matrix = np.array(dataset["labels"])
    total_evaluated = len(labels_matrix)    
    row_has_correct = labels_matrix.any(axis=1)
    upperbound_count = int(row_has_correct.sum())
    all_wrong_count = int((~row_has_correct).sum())    
    model_accuracies = labels_matrix.mean(axis=0)    
    model_names = [candidate["source"] for candidate in dataset["candidates"][0]]
    
    return {
        "total": total_evaluated,
        "all_wrong_cases": all_wrong_count,       
        "upperbound_accuracy": round(upperbound_count / total_evaluated, 4),
        "model_accuracies": {
            name: round(acc, 4) 
            for name, acc in zip(model_names, model_accuracies)
        }
    }

def _build_sql_agreement_dataset(gold_json_fp: str, models: List[str], preds_dir: str, 
                                database_root: str,
                                * ,
                                set_semantic: bool = True, dialect: str = "sqlite", 
                                timeout: float = 5.0, pred_file_suffix = "_predictions.jsonl",
                                instance_dir: str | None = None, instance_layout: str = "question"):
    gold_data = pd.read_json(gold_json_fp)
    if "question_id" not in gold_data.columns:
        gold_data["question_id"] = range(len(gold_data))  
    gold_data = gold_data.set_index("question_id")
    gold_data = gold_data.rename(columns={"SQL": "gold", "query": "gold"})
    if "sql" in gold_data.columns:
        gold_data = gold_data.drop(columns=["sql"])
    
    for model in models:
        pred_fp = Path(preds_dir) / f"{model}{pred_file_suffix}"
        pred_df = pd.read_json(pred_fp, lines=True).set_index("question_id")        
        gold_data[model] = pred_df["pred"]
    gold_data = gold_data.dropna(subset=models, how="all")    
    gold_data["models"] = [list(models) for _ in range(len(gold_data))]
    gold_data["candidates"] = gold_data[models].values.tolist()
    builder = SQLDatasetBuilder(
        database_root=database_root,
        dialect=dialect,
        timeout=timeout,
        set_semantic=set_semantic,
        instance_root=instance_dir,
        instance_layout=instance_layout,
    )
    ds = builder.build(gold_data)
    
    return ds

def build_sql_agreement_dataset(dataset_name: str, models: List[str], preds_dir: str, 
                                output_dir: str, * ,
                                set_semantic: bool = True, dialect: str = "sqlite", 
                                timeout: float = 5.0, pred_file_suffix = "_predictions.jsonl",
                                databases: Literal["official", "parseval", "testsuite"] = "official"):
    """``databases`` picks where queries are compared: "official" runs them on the official
    database; "parseval" on the original database (Spider's rebuilt on the aligned schema,
    scripts/align_spider_databases.py) and the ParSEval/hand-built databases of their question
    (scripts/parseval_instances.py, scripts/build_manual_instances.py), Spider and BIRD-dev only; "testsuite" on the
    official database and the test-suite databases of its database (Spider's official generator), Spider only."""
    if databases not in ("official", "parseval", "testsuite"):
        raise ValueError(f"databases must be 'official', 'parseval' or 'testsuite', not {databases!r}")
    if databases == "testsuite" and not dataset_name.startswith("spider"):
        raise ValueError(f"{dataset_name} has no test-suite databases")
    split = "train" if "train" in dataset_name else "test"
    instance_dir = None
    instance_layout = "database" if databases == "testsuite" else "question"
    if dataset_name == "bird_dev":
        dev_fp = os.environ["BIRD_DEV_FP"]        
        database_dir = os.environ["BIRD_DEV_DATABASE_DIR"]
        if databases == "parseval":
            instance_dir = os.environ["BIRD_DEV_INSTANCES_DIR"]
    elif dataset_name == "bird_train":
        if databases == "parseval":
            raise ValueError("bird_train has no ParSEval databases")
        dev_fp = os.environ["BIRD_TRAIN_FP"]        
        database_dir = os.environ["BIRD_TRAIN_DATABASE_DIR"]
    elif dataset_name == "spider_dev":
        dev_fp = os.environ["SPIDER_DEV_FP"]        
        database_dir = os.environ["SPIDER_DEV_DATABASE_DIR"]
        if databases == "parseval":
            database_dir = os.environ["SPIDER_ALIGNED_DATABASE_DIR"]
            instance_dir = os.environ["SPIDER_DEV_INSTANCES_DIR"]
        elif databases == "testsuite":
            instance_dir = os.environ["SPIDER_DEV_TESTSUITE_DIR"]
    elif dataset_name == "spider_test":
        dev_fp = os.environ["SPIDER_TEST_FP"]        
        database_dir = os.environ["SPIDER_TEST_DATABASE_DIR"]
        if databases == "parseval":
            database_dir = os.environ["SPIDER_ALIGNED_DATABASE_DIR"]
            instance_dir = os.environ["SPIDER_TEST_INSTANCES_DIR"]
        elif databases == "testsuite":
            instance_dir = os.environ["SPIDER_TEST_TESTSUITE_DIR"]
        
    ss = ABBREVIATIONS["set_semantic"] if set_semantic else "bs"    
    pv = {"official": "", "parseval": "_pv", "testsuite": "_ts"}[databases]
    output_ds_name = f"{dataset_name}_agr_q_sql_af_{''.join([ABBREVIATIONS[m] for m in models])}_{ss}{pv}_m{len(models)}"
    
    ds = _build_sql_agreement_dataset(
        gold_json_fp=dev_fp, 
        models=models, 
        preds_dir=preds_dir, 
        database_root=database_dir,
        set_semantic=set_semantic,
        dialect=dialect,
        timeout=timeout,
        pred_file_suffix=pred_file_suffix,
        instance_dir=instance_dir,
        instance_layout=instance_layout,
    )
    manifest = compute_dataset_metrics(ds)
    manifest["timeout"] = timeout
    manifest["databases"] = databases
    manifest["database_dir"] = database_dir
    manifest["instance_dir"] = instance_dir
    manifest["set_semantic"] = set_semantic
    DatasetDict({
        split : ds
    }).save_to_disk(Path(output_dir) / output_ds_name)
    
    with open(Path(output_dir) / output_ds_name / f"manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)
    
def build_prep_agreement_dataset():
    ...
    
# build_sql_agreement_dataset(
#     "bird_train", models=["Arctic-R1-7B", "QWen2.5-7B", "XiYanSQL-7B", "natural-sql-6.7B", "rnj-1-8B", "SQLCoder-7B", "OmniSQL-7B"], preds_dir="datasets/raw/preds/bird/train/csc/20260625_021204", output_dir = "datasets/processed/bird_train")

set_semantic = False

build_sql_agreement_dataset(
    "bird_dev", models=["Arctic-R1-7B", "QWen2.5-7B", "XiYanSQL-7B",  "rnj-1-8B", "OmniSQL-7B", "SQLCoder-7B", "natural-sql-6.7B"], preds_dir="datasets/raw/preds/bird/dev/dail/pred", output_dir = "datasets/processed/bird", pred_file_suffix = "_predictions.jsonl", set_semantic=set_semantic, databases="official")

build_sql_agreement_dataset(
    "spider_test", models=["Arctic-R1-7B", "QWen2.5-7B", "XiYanSQL-7B",  "rnj-1-8B", "OmniSQL-7B", "SQLCoder-7B", "natural-sql-6.7B"], preds_dir="datasets/raw/preds/spider/0shot", output_dir = "datasets/processed/spider", pred_file_suffix = "_test_predictions.jsonl", set_semantic=set_semantic, databases="official")


build_sql_agreement_dataset(
    "spider_dev", models=["Arctic-R1-7B", "QWen2.5-7B", "XiYanSQL-7B",  "rnj-1-8B", "OmniSQL-7B", "SQLCoder-7B", "natural-sql-6.7B"], preds_dir="datasets/raw/preds/spider/0shot", output_dir = "datasets/processed/spider", pred_file_suffix = "_predictions.jsonl", set_semantic=set_semantic, databases="official")