# Data Model Change Tracker

Track all data model changes discovered during spec reviews. Each change should be recorded here when identified, then marked as resolved once the corresponding spec is updated.

## How to Use

1. During a spec review, when a data model change is identified, add a row to the relevant table below
2. Set **Change Type** to one of: `add field`, `remove field`, `rename`, `change type`, `restructure`, `new file`, `remove file`
3. Set **Status** to one of: `identified`, `approved`, `applied`, `rejected`
4. After updating the spec, change status to `applied` and note the commit or date

---

## Input Schemas

Changes to input file formats, directory structure, and entity definitions.

| Spec File | Change Type | Description | Status |
| --------- | ----------- | ----------- | ------ |

## Internal Structures

Changes to in-memory representations, binary formats, and serialization.

| Spec File | Change Type | Description | Status |
| --------- | ----------- | ----------- | ------ |

## Output Formats

Changes to output schemas, Parquet layouts, and output infrastructure.

| Spec File | Change Type | Description | Status |
| --------- | ----------- | ----------- | ------ |

## Penalty System

Changes to penalty types, cascade logic, and override resolution.

| Spec File | Change Type | Description | Status |
| --------- | ----------- | ----------- | ------ |
