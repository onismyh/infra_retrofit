#!/usr/bin/env python3
"""Validate input data files for coal plant retrofit optimization."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# China bounds
LAT_MIN, LAT_MAX = 18, 54
LON_MIN, LON_MAX = 73, 136

def validate_coordinates(df, name, lat_col='latitude', lon_col='longitude'):
    """Validate lat/lon within China bounds."""
    issues = []
    if lat_col in df.columns and lon_col in df.columns:
        invalid_lat = df[(df[lat_col] < LAT_MIN) | (df[lat_col] > LAT_MAX)]
        invalid_lon = df[(df[lon_col] < LON_MIN) | (df[lon_col] > LON_MAX)]
        if len(invalid_lat) > 0:
            issues.append(f"{name}: {len(invalid_lat)} rows with invalid latitude")
        if len(invalid_lon) > 0:
            issues.append(f"{name}: {len(invalid_lon)} rows with invalid longitude")
        missing_coords = df[df[[lat_col, lon_col]].isna().any(axis=1)]
        if len(missing_coords) > 0:
            issues.append(f"{name}: {len(missing_coords)} rows with missing coordinates")
    return issues

def validate_capacity_consistency(plants_hub, plants_unit):
    """Check hub capacity matches sum of unit capacities."""
    issues = []
    for _, hub in plants_hub.iterrows():
        hub_id = hub['hub_id']
        hub_total = hub['total_capacity_mw']
        cooling_sum = hub['capacity_mw_once_through'] + hub['capacity_mw_recirculating'] + hub['capacity_mw_air']
        if not np.isclose(hub_total, cooling_sum, rtol=0.01):
            issues.append(f"Hub {hub_id}: total={hub_total:.1f} != cooling_sum={cooling_sum:.1f}")
    return issues

def validate_link_counts(links_df, name, hub_col='hub_id'):
    """Check link counts per hub."""
    issues = []
    if hub_col in links_df.columns:
        counts = links_df[hub_col].value_counts()
        low = counts[counts < 5]
        high = counts[counts > 500]
        if len(low) > 0:
            issues.append(f"{name}: {len(low)} hubs with <5 links")
        if len(high) > 0:
            issues.append(f"{name}: {len(high)} hubs with >500 links")
    return issues

def validate_cost_outliers(df, name, cost_cols):
    """Flag costs outside 3 std devs."""
    issues = []
    for col in cost_cols:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                mean, std = data.mean(), data.std()
                outliers = df[(df[col] < mean - 3*std) | (df[col] > mean + 3*std)]
                if len(outliers) > 0:
                    issues.append(f"{name}.{col}: {len(outliers)} outliers")
                invalid = df[df[col] <= 0]
                if len(invalid) > 0:
                    issues.append(f"{name}.{col}: {len(invalid)} non-positive values")
    return issues

def validate_missing_values(df, name, critical_cols):
    """Check for missing values in critical columns."""
    issues = []
    for col in critical_cols:
        if col not in df.columns:
            issues.append(f"{name}: missing column '{col}'")
        else:
            missing = df[col].isna().sum()
            if missing > 0:
                issues.append(f"{name}.{col}: {missing} missing values")
    return issues

def main():
    base_path = Path(__file__).parent.parent / "inputs"
    all_issues = []
    critical_failures = []

    print("=" * 60)
    print("INPUT DATA VALIDATION")
    print("=" * 60)

    # Validate plants
    try:
        plants_unit = pd.read_csv(base_path / "plants_unit.csv")
        plants_hub_100 = pd.read_csv(base_path / "plants_hub_100.csv")

        all_issues.extend(validate_coordinates(plants_unit, "plants_unit", 'latitude', 'longitude'))
        all_issues.extend(validate_coordinates(plants_hub_100, "plants_hub_100", 'centroid_latitude', 'centroid_longitude'))
        all_issues.extend(validate_capacity_consistency(plants_hub_100, plants_unit))
        all_issues.extend(validate_missing_values(plants_hub_100, "plants_hub_100",
                                                   ['hub_id', 'total_capacity_mw', 'centroid_latitude', 'centroid_longitude']))
    except Exception as e:
        critical_failures.append(f"Plants validation failed: {e}")

    # Validate ammonia supply
    try:
        ammonia_curve = pd.read_csv(base_path / "ammonia_supply_curve.csv")
        ammonia_links = pd.read_csv(base_path / "ammonia_supply_links_100.csv")

        all_issues.extend(validate_coordinates(ammonia_curve, "ammonia_supply_curve"))
        all_issues.extend(validate_link_counts(ammonia_links, "ammonia_supply_links_100"))
        all_issues.extend(validate_cost_outliers(ammonia_curve, "ammonia_supply_curve",
                                                  ['weighted_lcoh_usd_per_kg_h2', 'nh3_cost_lb_usd_per_kg']))
    except Exception as e:
        critical_failures.append(f"Ammonia validation failed: {e}")

    # Validate biomass supply
    try:
        biomass_curve = pd.read_csv(base_path / "biomass_supply_curve.csv")
        biomass_links = pd.read_csv(base_path / "biomass_supply_links_100.csv")

        all_issues.extend(validate_coordinates(biomass_curve, "biomass_supply_curve"))
        all_issues.extend(validate_link_counts(biomass_links, "biomass_supply_links_100"))
        all_issues.extend(validate_cost_outliers(biomass_curve, "biomass_supply_curve", ['cost_usd_per_kg']))
    except Exception as e:
        critical_failures.append(f"Biomass validation failed: {e}")

    # Validate water supply
    try:
        water_nodes = pd.read_csv(base_path / "water_nodes.csv")
        water_links = pd.read_csv(base_path / "water_supply_links_100.csv")

        all_issues.extend(validate_coordinates(water_nodes, "water_nodes"))
        all_issues.extend(validate_link_counts(water_links, "water_supply_links_100"))
    except Exception as e:
        critical_failures.append(f"Water validation failed: {e}")

    # Validate storage hubs
    try:
        storage = pd.read_csv(base_path / "storage_hubs.csv")
        all_issues.extend(validate_coordinates(storage, "storage_hubs"))
        all_issues.extend(validate_missing_values(storage, "storage_hubs",
                                                   ['hub_id', 'latitude', 'longitude', 'storage_capacity_kg']))
    except Exception as e:
        critical_failures.append(f"Storage validation failed: {e}")

    # Print summary
    print(f"\nValidation Results:")
    print(f"  Critical failures: {len(critical_failures)}")
    print(f"  Warnings/issues: {len(all_issues)}")

    # Save report
    report_path = base_path / "validation_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("INPUT DATA VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        if critical_failures:
            f.write("CRITICAL FAILURES:\n")
            for issue in critical_failures:
                f.write(f"  [FAIL] {issue}\n")
            f.write("\n")

        if all_issues:
            f.write("WARNINGS:\n")
            for issue in all_issues:
                f.write(f"  [WARN] {issue}\n")
        else:
            f.write("All checks passed.\n")

    print(f"\nDetailed report saved to: {report_path}")

    if critical_failures:
        print("\n[FAIL] VALIDATION FAILED")
        return 1
    elif all_issues:
        print("\n[WARN] VALIDATION PASSED WITH WARNINGS")
        return 0
    else:
        print("\n[PASS] VALIDATION PASSED")
        return 0

if __name__ == "__main__":
    sys.exit(main())
