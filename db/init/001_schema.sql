CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS simulation_batches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_name TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID REFERENCES simulation_batches(id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    solver_name TEXT NOT NULL DEFAULT 'dolfinx',
    mesh_nx INTEGER NOT NULL,
    mesh_ny INTEGER NOT NULL,
    duration_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS simulation_inputs (
    run_id UUID PRIMARY KEY REFERENCES simulation_runs(id) ON DELETE CASCADE,
    length_m DOUBLE PRECISION NOT NULL,
    height_m DOUBLE PRECISION NOT NULL,
    young_modulus_pa DOUBLE PRECISION NOT NULL,
    poisson_ratio DOUBLE PRECISION NOT NULL,
    traction_pa DOUBLE PRECISION NOT NULL,
    boundary_condition TEXT NOT NULL DEFAULT 'left_clamped',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS simulation_outputs (
    run_id UUID PRIMARY KEY REFERENCES simulation_runs(id) ON DELETE CASCADE,
    max_displacement_m DOUBLE PRECISION NOT NULL,
    max_von_mises_pa DOUBLE PRECISION NOT NULL,
    raw_artifact_uri TEXT,
    processed_artifact_uri TEXT,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS simulation_features (
    run_id UUID NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    feature_value DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, feature_name)
);

CREATE TABLE IF NOT EXISTS ml_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifact_uri TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_registry (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID NOT NULL REFERENCES ml_experiments(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    model_version INTEGER NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('staging', 'production', 'archived')),
    promoted_at TIMESTAMPTZ,
    UNIQUE (model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_simulation_runs_created_at ON simulation_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_simulation_runs_status ON simulation_runs(status);
CREATE INDEX IF NOT EXISTS idx_simulation_features_name ON simulation_features(feature_name);
CREATE INDEX IF NOT EXISTS idx_ml_experiments_created_at ON ml_experiments(created_at);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'geometry_type_enum') THEN
        CREATE TYPE geometry_type_enum AS ENUM ('with_hole', 'without_hole', 'with_hole_moving');
    END IF;
END$$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'geometry_type_enum') THEN
        IF NOT EXISTS (
            SELECT 1
            FROM pg_enum e
            JOIN pg_type t ON t.oid = e.enumtypid
            WHERE t.typname = 'geometry_type_enum' AND e.enumlabel = 'with_hole_moving'
        ) THEN
            ALTER TYPE geometry_type_enum ADD VALUE 'with_hole_moving';
        END IF;
    END IF;
END$$;

CREATE TABLE IF NOT EXISTS simulation_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    simulation_id UUID NOT NULL UNIQUE,
    ts TIMESTAMPTZ NOT NULL,
    geometry_type geometry_type_enum NOT NULL,
    material_category TEXT NOT NULL,
    dimension_category TEXT NOT NULL,
    length_m DOUBLE PRECISION NOT NULL CHECK (length_m > 0),
    height_m DOUBLE PRECISION NOT NULL CHECK (height_m > 0),
    young_modulus_pa DOUBLE PRECISION NOT NULL CHECK (young_modulus_pa > 0),
    poisson_ratio DOUBLE PRECISION NOT NULL CHECK (poisson_ratio > 0 AND poisson_ratio < 0.5),
    traction_pa DOUBLE PRECISION NOT NULL CHECK (traction_pa >= 0),
    mesh_nx INTEGER NOT NULL CHECK (mesh_nx >= 8),
    mesh_ny INTEGER NOT NULL CHECK (mesh_ny >= 4),
    hole_radius_ratio DOUBLE PRECISION CHECK (hole_radius_ratio > 0 AND hole_radius_ratio < 0.5),
    hole_cx_ratio DOUBLE PRECISION CHECK (hole_cx_ratio > 0 AND hole_cx_ratio < 1),
    hole_cy_ratio DOUBLE PRECISION CHECK (hole_cy_ratio > 0 AND hole_cy_ratio < 1),
    max_displacement_m DOUBLE PRECISION NOT NULL CHECK (max_displacement_m >= 0),
    max_von_mises_pa DOUBLE PRECISION NOT NULL CHECK (max_von_mises_pa >= 0),
    solver_name TEXT NOT NULL,
    solver_version TEXT NOT NULL,
    data_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (
            geometry_type = 'with_hole'
            AND hole_radius_ratio IS NOT NULL
            AND hole_cx_ratio IS NULL
            AND hole_cy_ratio IS NULL
        )
        OR (
            geometry_type = 'with_hole_moving'
            AND hole_radius_ratio IS NOT NULL
            AND hole_cx_ratio IS NOT NULL
            AND hole_cy_ratio IS NOT NULL
        )
        OR (
            geometry_type = 'without_hole'
            AND hole_radius_ratio IS NULL
            AND hole_cx_ratio IS NULL
            AND hole_cy_ratio IS NULL
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_simulation_records_geometry_type ON simulation_records(geometry_type);
CREATE INDEX IF NOT EXISTS idx_simulation_records_data_version ON simulation_records(data_version);
CREATE INDEX IF NOT EXISTS idx_simulation_records_ts ON simulation_records(ts);
