import numpy as np
import json
import os
import zipfile
import argparse
from pathlib import Path


def convert_to_serializable(obj):
    """
    Recursively convert numpy types to standard Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bytes):
        return obj.decode("utf-8")
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def convert_database(database_path, output_path):
    print(f"Loading database from: {database_path}")

    if not os.path.exists(database_path):
        print(f"Error: File {database_path} not found.")
        return

    # Load all data from the existing npz
    with np.load(database_path) as data:
        # Create a dictionary of the arrays so we can modify it
        output_arrays = dict(data)

        # Convert pattern_largest_edge (if included) to float32s
        if "pattern_largest_edge" in output_arrays:
            output_arrays["pattern_largest_edge"] = output_arrays["pattern_largest_edge"].astype(np.float32)

        # Convert properties to json
        if "props_packed" not in output_arrays:
            print("Error: 'props_packed' key not found in the .npz file.")
            return

        props_packed = output_arrays["props_packed"]

        # Initialize dictionary based on Tetra3 class structure
        db_props = {
            "pattern_mode": None,
            "hash_table_type": None,
            "pattern_size": None,
            "pattern_bins": None,
            "pattern_max_error": None,
            "max_fov": None,
            "min_fov": None,
            "star_catalog": None,
            "epoch_equinox": None,
            "epoch_proper_motion": None,
            "lattice_field_oversampling": None,
            "patterns_per_lattice_field": None,
            "verification_stars_per_fov": None,
            "star_max_magnitude": None,
            "range_ra": None,
            "range_dec": None,
            "presort_patterns": None,
            "num_patterns": None,
        }

        print("-" * 40)
        print("Extracting Properties from props_packed:")
        print("-" * 40)

        for key in db_props.keys():
            try:
                if key in props_packed.dtype.names:
                    val = props_packed[key][()]
                    print(f"Property: {key:<30} | Value: {val}")
                    db_props[key] = val
                else:
                    raise ValueError("Key not directly found")

            except (ValueError, KeyError):
                # Handle legacy remappings
                if key == "verification_stars_per_fov":
                    if "catalog_stars_per_fov" in props_packed.dtype.names:
                        val = props_packed["catalog_stars_per_fov"][()]
                        print(
                            f"Property: {key:<30} | Value: {val} (mapped from 'catalog_stars_per_fov')"
                        )
                        db_props[key] = val

                elif key == "star_max_magnitude":
                    if "star_min_magnitude" in props_packed.dtype.names:
                        val = props_packed["star_min_magnitude"][()]
                        print(
                            f"Property: {key:<30} | Value: {val} (mapped from 'star_min_magnitude')"
                        )
                        db_props[key] = val

                elif key == "presort_patterns":
                    val = False
                    print(f"Property: {key:<30} | Value: {val} (Defaulted)")
                    db_props[key] = val

                elif key == "star_catalog":
                    val = "unknown"
                    print(f"Property: {key:<30} | Value: {val} (Defaulted)")
                    db_props[key] = val

                elif key == "num_patterns":
                    if "pattern_catalog" in output_arrays:
                        val = output_arrays["pattern_catalog"].shape[0] // 2
                        print(
                            f"Property: {key:<30} | Value: {val} (Calculated from catalog size)"
                        )
                        db_props[key] = val
                    else:
                        print(
                            f"Property: {key:<30} | Value: None (Warning: pattern_catalog missing)"
                        )
                        db_props[key] = None

                else:
                    print(f"Property: {key:<30} | Value: None (Missing)")
                    db_props[key] = None

        if db_props["min_fov"] is None:
            db_props["min_fov"] = db_props["max_fov"]
            print(
                f"Property: {'min_fov':<30} | Value: {db_props['min_fov']} (Copied from max_fov)"
            )

        print("-" * 40)

        # Serialize to JSON string
        serializable_props = convert_to_serializable(db_props)
        json_str = json.dumps(serializable_props, indent=4)
        json_str = json_str.replace('NaN', 'null')

        # REMOVE the old props_packed from the arrays to be saved via numpy
        del output_arrays["props_packed"]

        print(f"Saving numpy arrays to {output_path}...")
        # 1. Save the numpy arrays first.
        np.savez_compressed(output_path, **output_arrays)

        print(f"Appending properties.json to archive...")
        # 2. Open the existing NPZ (zip) in append mode and add the JSON file directly.
        with zipfile.ZipFile(output_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("properties.json", json_str)

        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Tetra3 props_packed.npy to properties.json within an .npz file."
    )

    # Named arguments
    parser.add_argument(
        "--input",
        "-i",
        default="default_database.npz",
        help="Input .npz file path (default: default_database.npz)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="default_database_json.npz",
        help="Output .npz file path (default: default_database_json.npz)",
    )

    args = parser.parse_args()

    convert_database(database_path=args.input, output_path=args.output)
