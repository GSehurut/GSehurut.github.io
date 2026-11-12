---
layout: post
title: "Working With Microsoft Fabric for Actuarial Data: Lessons From Practice"
date: 2026-10-19
description: "Lakehouse or warehouse, medallion layers, idempotent PySpark merges, Delta table maintenance, Git integration and capacity: practical lessons for running actuarial pipelines on Microsoft Fabric."
categories: [Data Engineering, Microsoft Fabric]
tags: [Microsoft Fabric, PySpark, Delta Lake, Lakehouse, Power BI, Data Engineering]
---

Microsoft Fabric brings data engineering, warehousing and Power BI into a single platform built around one data lake, OneLake. For actuarial and insurance teams that already live in the Microsoft ecosystem, that is an attractive proposition: reserving data, pricing datasets and executive dashboards can share the same storage and governance instead of being copied between tools.

This post covers the decisions and habits that have mattered most when moving actuarial pipelines, such as reserving and loss triangle data, from locally run scripts onto scheduled Fabric pipelines.

## The building blocks, briefly

- **OneLake** is a single logical data lake for the whole organisation. Every workspace stores its data there, in open Delta Parquet format.
- **Lakehouse** items hold files and Delta tables, and are usually worked on with Spark notebooks. Each one comes with a read only **SQL analytics endpoint** for T-SQL queries.
- **Warehouse** items offer a full read and write T-SQL experience over the same Delta format.
- **Data pipelines** (from Data Factory) orchestrate copies, notebooks and stored procedures on a schedule.
- **Semantic models** in **Direct Lake** mode let Power BI read Delta tables directly from OneLake without importing a copy.
- **Shortcuts** let one lakehouse reference data in another location without duplicating it.

## Lakehouse or warehouse?

Both store Delta tables in OneLake, so this is mainly a question of how your team works:

| If your team mostly... | Lean towards |
| --- | --- |
| writes Python and PySpark, works with files and semi structured data | Lakehouse |
| writes T-SQL, wants stored procedures and multi table transactions | Warehouse |
| needs both | Lakehouse for ingestion and transformation, Warehouse or SQL endpoint for serving |

For actuarial teams with a mix of Python and SQL skills, a lakehouse for ingestion and transformation with SQL access for analysts has worked well.

## Medallion layers with a purpose

Microsoft's guidance recommends a medallion architecture: **bronze** for raw data as received, **silver** for cleaned and conformed data, **gold** for business ready models. The pattern is only useful if each layer has a clear contract:

- **Bronze:** files exactly as received (claims extracts, bordereaux, system exports), plus load metadata. Never edited.
- **Silver:** validated, typed and deduplicated, at a documented grain, with consistent keys and currencies.
- **Gold:** dimensional models and aggregates shaped for reporting: claim movement facts, triangles, premium by class.

Keeping bronze immutable means that when a transformation bug is found, you can fix it and replay history rather than asking source teams for files again.

## Idempotent loads with Delta MERGE

Scheduled pipelines fail, get rerun, and receive corrected files. Every load from bronze to silver should be safe to run twice. Delta Lake's `MERGE` makes this straightforward in a Fabric notebook:

```python
from pyspark.sql import functions as F
from delta.tables import DeltaTable

valuation_date = "2026-09-30"  # passed in as a pipeline parameter in practice

incoming = (
    spark.read.option("header", True)
    .csv(f"Files/bronze/claims/{valuation_date}/*.csv")
    .withColumn("paid_to_date", F.col("paid_to_date").cast("decimal(18,2)"))
    .withColumn("outstanding", F.col("outstanding").cast("decimal(18,2)"))
    .withColumn("valuation_date", F.lit(valuation_date).cast("date"))
    .withColumn("_loaded_at", F.current_timestamp())
    .dropDuplicates(["claim_id", "valuation_date"])
)

target = DeltaTable.forName(spark, "silver_claim_snapshot")

(
    target.alias("t")
    .merge(
        incoming.alias("s"),
        "t.claim_id = s.claim_id AND t.valuation_date = s.valuation_date",
    )
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
)
```

Rerunning for the same valuation date updates rows rather than duplicating them. The same pattern, keyed on the documented grain, works for premium and exposure data.

## Validate before you merge

A merge will happily write bad data. Add checks between reading and merging, and fail the notebook with a clear message so the pipeline run shows as failed:

```python
problems = []
if incoming.filter(F.col("claim_id").isNull()).limit(1).count() > 0:
    problems.append("null claim_id")
if incoming.filter(F.col("paid_to_date") < 0).limit(1).count() > 0:
    problems.append("negative paid_to_date")

if problems:
    raise ValueError(f"Claims load {valuation_date} rejected: {', '.join(problems)}")
```

## Look after your Delta tables

Delta tables need some housekeeping to stay fast:

- **Small files** build up when you append little batches frequently. Run `OPTIMIZE` periodically to compact them.
- **Old versions** accumulate for time travel. `VACUUM` removes files no longer needed beyond your retention period. Keep retention long enough for any "as at" queries you rely on.
- **V-Order**, Fabric's write time optimisation for Parquet, improves read performance for Power BI and SQL. It is worth understanding whether it is enabled for your workloads.
- **Partition sparingly.** Partitioning a modest reserving table by date can create thousands of tiny files and make things slower. Partition only large tables, on columns you actually filter by.

The Delta Lake paper by Armbrust et al. (2020) explains why these operations matter: the transaction log and file layout are what give Delta its reliability and performance.

## Treat workspaces like environments

Fabric supports **Git integration** (Azure DevOps or GitHub) for workspace items and **deployment pipelines** to promote content between development, test and production workspaces. Use them:

- Develop in a dev workspace connected to a feature branch.
- Review notebook and pipeline changes through pull requests like any other code.
- Promote to production through a deployment pipeline, with parameters (lakehouse names, paths) swapped per stage.

Keeping business logic in version controlled notebooks or Python modules, rather than in manual edits in production, is what makes a pipeline auditable.

## Watch your capacity

Fabric runs on capacities, and every workload (Spark, SQL, Power BI refreshes) draws from the same pool. A heavy notebook scheduled at the same time as the morning dashboard refresh can slow everything down. The **Fabric Capacity Metrics** app shows where capacity goes. Stagger schedules, right size Spark sessions, and avoid rerunning full history when an incremental load would do.

## Serving with Direct Lake

Once gold tables are in good shape, a Direct Lake semantic model lets Power BI read them straight from OneLake. Dashboards see fresh data without a separate import refresh, and measures are defined once in the semantic model rather than in every report. Gold tables designed as a clean star schema (see the data modelling post) make Direct Lake models far simpler.

## Lessons in short

1. Keep bronze immutable and replayable.
2. Give each medallion layer a written contract.
3. Make every load idempotent with `MERGE` on the documented grain.
4. Validate before writing, and fail loudly.
5. Maintain Delta tables with `OPTIMIZE` and a considered `VACUUM` policy.
6. Use Git integration and deployment pipelines; no hand edits in production.
7. Monitor capacity and schedule workloads with care.

## References

- Microsoft Learn. *What is Microsoft Fabric?* learn.microsoft.com/fabric
- Microsoft Learn. *Implement medallion lakehouse architecture in Microsoft Fabric*. learn.microsoft.com/fabric
- Microsoft Learn. *Direct Lake overview*. learn.microsoft.com/fabric
- Microsoft Learn. *Introduction to Git integration* and *Introduction to deployment pipelines* (Fabric lifecycle management). learn.microsoft.com/fabric
- Armbrust, M. et al. (2020). Delta Lake: high-performance ACID table storage over cloud object stores. *Proceedings of the VLDB Endowment*, 13(12), 3411 to 3424.
- Delta Lake documentation. docs.delta.io
