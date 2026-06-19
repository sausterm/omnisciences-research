"""
Parsers for 1D proton potential energy scans from DFT calculations.

Supports:
    - Gaussian relaxed scan (.log)
    - ORCA relaxed scan (.out)
    - Generic CSV/TSV (distance, energy columns)
    - NumPy .npy / .npz files

All parsers return (r_grid, potential) where:
    r_grid: 1D array of proton coordinates in angstrom
    potential: 1D array of potential energy in eV
"""

import numpy as np
from pathlib import Path


def parse_scan_csv(
    filepath: str,
    r_col: int = 0,
    e_col: int = 1,
    r_unit: str = "angstrom",
    e_unit: str = "eV",
    delimiter: str | None = None,
    skip_header: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a CSV/TSV file with distance and energy columns.

    Args:
        filepath: Path to the file.
        r_col: Column index for distance (0-based).
        e_col: Column index for energy (0-based).
        r_unit: Unit of distance ('angstrom' or 'bohr').
        e_unit: Unit of energy ('eV', 'hartree', 'kcal/mol', 'kJ/mol', 'cm-1').
        delimiter: Column delimiter (None = whitespace).
        skip_header: Number of header lines to skip.

    Returns:
        (r_grid, potential) in angstrom and eV.
    """
    data = np.loadtxt(filepath, delimiter=delimiter, skiprows=skip_header)
    r = data[:, r_col]
    e = data[:, e_col]

    r = _convert_distance(r, r_unit)
    e = _convert_energy(e, e_unit)

    # Sort by distance
    order = np.argsort(r)
    return r[order], e[order]


def parse_scan_gaussian(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Gaussian relaxed scan .log file.

    Extracts the scan coordinate and SCF energies from lines like:
        Summary of Optimized Potential Surface Scan
    or from individual step energies.

    Args:
        filepath: Path to Gaussian .log file.

    Returns:
        (r_grid, potential) in angstrom and eV.
    """
    HARTREE_TO_EV = 27.211386245988

    distances = []
    energies = []

    with open(filepath) as f:
        lines = f.readlines()

    # Strategy 1: Look for "Summary of Optimized Potential Surface Scan"
    in_summary = False
    for i, line in enumerate(lines):
        if "Summary of Optimized Potential Surface Scan" in line:
            in_summary = True
            continue
        if in_summary:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    d = float(parts[0])
                    e = float(parts[1])
                    distances.append(d)
                    energies.append(e)
                except ValueError:
                    if distances:
                        break
                    continue

    # Strategy 2: Fallback to extracting from "Scan" steps
    if not distances:
        current_e = None
        current_d = None
        for line in lines:
            if "SCF Done" in line:
                parts = line.split()
                for j, p in enumerate(parts):
                    if p == "=":
                        try:
                            current_e = float(parts[j + 1])
                        except (ValueError, IndexError):
                            pass
                        break
            if "Scan" in line and "!" in line:
                parts = line.split()
                for j, p in enumerate(parts):
                    try:
                        val = float(p)
                        current_d = val
                        break
                    except ValueError:
                        continue
            if "-- Stationary point found" in line and current_e is not None and current_d is not None:
                distances.append(current_d)
                energies.append(current_e)
                current_e = None
                current_d = None

    if not distances:
        raise ValueError(f"Could not parse scan data from {filepath}")

    r = np.array(distances)
    e = np.array(energies) * HARTREE_TO_EV
    e -= e.min()

    order = np.argsort(r)
    return r[order], e[order]


def parse_scan_orca(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse an ORCA relaxed scan .out file.

    Looks for the scan summary table or individual scan step energies.

    Args:
        filepath: Path to ORCA .out file.

    Returns:
        (r_grid, potential) in angstrom and eV.
    """
    HARTREE_TO_EV = 27.211386245988

    distances = []
    energies = []

    with open(filepath) as f:
        lines = f.readlines()

    # Look for "The Calculated Surface using the SCF energy"
    for i, line in enumerate(lines):
        if "The Calculated Surface using the" in line:
            # Skip header lines
            j = i + 1
            while j < len(lines):
                parts = lines[j].split()
                if len(parts) >= 2:
                    try:
                        d = float(parts[0])
                        e = float(parts[1])
                        distances.append(d)
                        energies.append(e)
                    except ValueError:
                        if distances:
                            break
                j += 1

    # Fallback: extract from RELAXED SURFACE SCAN STEP
    if not distances:
        current_e = None
        scan_coord = None
        for line in lines:
            if "FINAL SINGLE POINT ENERGY" in line:
                parts = line.split()
                try:
                    current_e = float(parts[-1])
                except ValueError:
                    pass
            if "Setting up the" in line and "scan coordinate" in line.lower():
                parts = line.split()
                for p in parts:
                    try:
                        scan_coord = float(p)
                    except ValueError:
                        continue
            if "RELAXED SURFACE SCAN STEP" in line and current_e is not None:
                if scan_coord is not None:
                    distances.append(scan_coord)
                else:
                    distances.append(len(energies))
                energies.append(current_e)

    if not distances:
        raise ValueError(f"Could not parse scan data from {filepath}")

    r = np.array(distances)
    e = np.array(energies) * HARTREE_TO_EV
    e -= e.min()

    order = np.argsort(r)
    return r[order], e[order]


def parse_scan_numpy(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a scan from a NumPy file.

    For .npy: expects shape (N, 2) with columns [distance, energy].
    For .npz: expects keys 'r' and 'V' (or 'distance' and 'energy').

    Distance in angstrom, energy in eV.

    Args:
        filepath: Path to .npy or .npz file.

    Returns:
        (r_grid, potential) in angstrom and eV.
    """
    p = Path(filepath)
    if p.suffix == ".npz":
        data = np.load(filepath)
        if "r" in data and "V" in data:
            r, e = data["r"], data["V"]
        elif "distance" in data and "energy" in data:
            r, e = data["distance"], data["energy"]
        else:
            raise ValueError(f"NPZ file must have keys 'r'/'V' or 'distance'/'energy', got {list(data.keys())}")
    else:
        data = np.load(filepath)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"NPY file must have shape (N, 2+), got {data.shape}")
        r, e = data[:, 0], data[:, 1]

    order = np.argsort(r)
    return r[order], e[order]


def parse_scan(filepath: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Auto-detect file format and parse a 1D potential energy scan.

    Args:
        filepath: Path to scan file.
        **kwargs: Additional arguments passed to the format-specific parser.

    Returns:
        (r_grid, potential) in angstrom and eV.
    """
    p = Path(filepath)
    suffix = p.suffix.lower()

    if suffix == ".log":
        return parse_scan_gaussian(filepath)
    elif suffix == ".out":
        return parse_scan_orca(filepath)
    elif suffix in (".npy", ".npz"):
        return parse_scan_numpy(filepath)
    elif suffix in (".csv", ".tsv", ".dat", ".txt"):
        if suffix == ".tsv":
            kwargs.setdefault("delimiter", "\t")
        return parse_scan_csv(filepath, **kwargs)
    else:
        # Try CSV as fallback
        return parse_scan_csv(filepath, **kwargs)


# =====================================================================
# Unit conversion helpers
# =====================================================================

def _convert_distance(r: np.ndarray, unit: str) -> np.ndarray:
    """Convert distance to angstrom."""
    if unit == "angstrom":
        return r
    elif unit == "bohr":
        return r * 0.529177249
    else:
        raise ValueError(f"Unknown distance unit: {unit}")


def _convert_energy(e: np.ndarray, unit: str) -> np.ndarray:
    """Convert energy to eV."""
    if unit == "eV":
        return e
    elif unit == "hartree":
        return e * 27.211386245988
    elif unit == "kcal/mol":
        return e / 23.0605
    elif unit == "kJ/mol":
        return e / 96.485
    elif unit == "cm-1":
        return e / 8065.544
    else:
        raise ValueError(f"Unknown energy unit: {unit}")
