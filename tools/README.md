# Powers-rs Tools

This directory contains utility tools for working with powers-rs.

## migrate_json.py

Migrates `recourse.json` files from the old temporal_model format to the new unified format introduced in v0.4.0.

### What it does

**For Independent models** (`{"type": "independent"}`):
- Extracts `seasonal_means` and `seasonal_stds` from `seasonal_distributions`
- Adds `ar_orders: [0, 0, ...]` and `ar_coefficients: [[], [], ...]`
- Removes the `"type"` field

**For PAR models** (`{"type": "periodic_ar"}`):
- Simply removes the `"type"` field
- Keeps all other fields unchanged

### Usage

```bash
# Dry run (show changes without modifying)
./tools/migrate_json.py examples/03-multistage/recourse.json -d

# Migrate and write to new file
./tools/migrate_json.py examples/03-multistage/recourse.json -o examples/03-multistage/recourse_new.json

# Migrate in place (overwrites original)
./tools/migrate_json.py examples/03-multistage/recourse.json -i

# Migrate with validation
./tools/migrate_json.py examples/03-multistage/recourse.json -i -v
```

### Options

- `input` - Input JSON file (required)
- `-o, --output` - Output JSON file (default: stdout)
- `-i, --in-place` - Modify file in place
- `-d, --dry-run` - Show changes without writing
- `-v, --validate` - Validate output JSON structure

### Examples

#### Migrate a single file

```bash
cd /path/to/powers
./tools/migrate_json.py examples/03-multistage/recourse.json -i -v
```

#### Migrate all examples

```bash
for file in examples/*/recourse.json; do
    echo "Migrating $file..."
    ./tools/migrate_json.py "$file" -i -v
done
```

#### Preview changes before migrating

```bash
./tools/migrate_json.py examples/03-multistage/recourse.json -d | head -50
```

### Before and After

**Before (Independent)**:
```json
{
  "temporal_model": {
    "type": "independent"
  },
  "seasonal_distributions": [
    {"season_id": 0, "distribution": {"type": "normal", "mean": 100.0, "std_dev": 20.0}}
  ]
}
```

**After (Independent)**:
```json
{
  "temporal_model": {
    "num_seasons": 1,
    "seasonal_means": [100.0],
    "seasonal_stds": [20.0],
    "ar_orders": [0],
    "ar_coefficients": [[]]
  },
  "seasonal_distributions": [
    {"season_id": 0, "distribution": {"type": "normal", "mean": 100.0, "std_dev": 20.0}}
  ]
}
```

**Before (PAR)**:
```json
{
  "temporal_model": {
    "type": "periodic_ar",
    "num_seasons": 12,
    "ar_orders": [1, 1, ...],
    "ar_coefficients": [[0.7], [0.7], ...],
    "seasonal_means": [70.0, 65.0, ...],
    "seasonal_stds": [20.0, 20.0, ...]
  }
}
```

**After (PAR)**:
```json
{
  "temporal_model": {
    "num_seasons": 12,
    "ar_orders": [1, 1, ...],
    "ar_coefficients": [[0.7], [0.7], ...],
    "seasonal_means": [70.0, 65.0, ...],
    "seasonal_stds": [20.0, 20.0, ...]
  }
}
```

### Validation

The script validates:
- `num_seasons` field is present
- Array lengths match `num_seasons`
- `ar_coefficients[i]` length matches `ar_orders[i]`

### Notes

- The old format still works (backward compatible)
- Migration is optional but recommended
- The tool preserves all other fields in the JSON
- LogNormal3 mean/std extraction uses correct formulas

### See Also

- `docs/json-schema-v2.md` - New JSON format specification
- `docs/migration-guide.md` - Complete migration guide
- `CHANGELOG.md` - v0.4.0 release notes
