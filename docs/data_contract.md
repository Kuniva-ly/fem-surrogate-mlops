# Data Contract: sim_v1 family

Each record corresponds to one simulation run.

Required fields:
- simulation_id (string UUID)
- timestamp (ISO-8601 UTC)
- material_category (non-empty string, e.g. steel/aluminum/custom)
- dimension_category (non-empty string, e.g. small/medium/custom)
- length_m (float > 0)
- height_m (float > 0)
- young_modulus_pa (float > 0)
- poisson_ratio (float in (0, 0.5))
- traction_pa (float >= 0)
- mesh_nx (int >= 8)
- mesh_ny (int >= 4)
- max_displacement_m (float >= 0)
- max_von_mises_pa (float >= 0)
- solver_name (string, default: dolfinx)
- solver_version (string)
- data_version (string, e.g. sim_v1)

Optional fields:
- hole_radius_ratio (float in (0, 0.5), present for `with_hole` and `with_hole_moving` datasets)
- hole_cx_ratio (float in (0, 1), present for `with_hole_moving` datasets)
- hole_cy_ratio (float in (0, 1), present for `with_hole_moving` datasets)
- geometry_type (string, recommended values: `with_hole`, `without_hole`, `with_hole_moving`)

Storage layout:
- data/raw/sim_v1/date=YYYY-MM-DD/part-*.parquet
- data/raw/sim_v1_without_hole/date=YYYY-MM-DD/part-*.parquet
- data/raw/sim_v2_moving_hole/date=YYYY-MM-DD/part-*.parquet
- data/processed/date=YYYY-MM-DD/{train,val,test}.parquet
- data/features/<feature_group>/<feature_version>/<geometry>/date=YYYY-MM-DD/features.parquet
- MinIO prefix recommendation:
  - raw-simulations/with_hole/sim_v1/date=YYYY-MM-DD/part-*.parquet
  - raw-simulations/without_hole/sim_v1_without_hole/date=YYYY-MM-DD/part-*.parquet
  - raw-simulations/with_hole_moving/sim_v2_moving_hole/date=YYYY-MM-DD/part-*.parquet
  - processed-simulations/with_hole/date=YYYY-MM-DD/{train,val,test}.parquet
  - features/stress_model/v1/with_hole/date=YYYY-MM-DD/features.parquet

Validation rules:
- No nulls on required fields
- simulation_id unique in a batch
- Numeric range checks as specified above
- Non-empty category labels for material_category and dimension_category

PostgreSQL mapping:
- Table cible: `simulation_records`
- Colonne temporelle: `timestamp` -> `ts`
- Upsert key: `simulation_id`
- `geometry_type` attendu: `with_hole`, `without_hole` ou `with_hole_moving`
