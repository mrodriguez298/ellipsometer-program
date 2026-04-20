# Ellipsometer-Program

A desktop GUI application for analyzing Thorlabs ellipsometer CSV exports.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](#)
[![UI](https://img.shields.io/badge/UI-Tkinter-green.svg)](#)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](#)

## Key Features

- Single-file analysis for Thorlabs WPH/WPQ and AHWP/AQWP waveplates
- Batch processing of multiple CSV exports with PASS/FAIL reporting
- Lu-Chipman Mueller matrix decomposition for each wavelength
- Retardance interpolation at the part's design wavelength
- Retardance-vs-wavelength plot for achromatic -340 parts
- CSV export of batch results

## What It Does

`ellipsometerProgram.py` reads an ellipsometer CSV export, drops unused columns, and parses 4×4 Mueller matrices from the file.

For single-wavelength waveplates (`WPH*` and `WPQ*`), the app:

- interpolates measured retardance to the design wavelength
- compares the result against Thorlabs tolerance
- displays a PASS/FAIL verdict

For achromatic `AHWP*-340` and `AQWP*-340` parts, the app:

- plots retardance as a function of wavelength
- supports the 260 nm to 410 nm range used by the instrument

## Project Structure

- `ellipsometerProgram.py` — main Tkinter application and signal-processing logic
- `README.md` — project documentation

## Requirements

- Python 3.10 or newer
- `numpy`
- `pandas`
- `matplotlib`
- `tkinter` (included with standard Python on Windows)

## Running the App

From the project directory, run:

```powershell
python ellipsometerProgram.py
```

The GUI opens with two modes:

- **Single File Mode** — select one CSV and a Thorlabs part number
- **Batch Mode** — select multiple CSV files and run validation for the selected part number

## Usage

### Single File Mode

1. Choose a supported Thorlabs part number from the dropdown.
2. Browse to the ellipsometer CSV file.
3. Click **Run Analysis**.
4. For single-wavelength waveplates, the app shows a PASS/FAIL result.
5. For `-340` achromatic parts, the app shows a retardance-vs-wavelength plot.

### Batch Mode

1. Choose a supported Thorlabs part number.
2. Select one or more CSV files.
3. Click **Run Batch**.
4. Review the file-by-file verdicts in the table.
5. Export the results to CSV with **Export CSV…**.

## Notes

- The app expects standard Thorlabs ellipsometer CSV exports with each row representing a wavelength and a 4×4 Mueller matrix.
- Unsupported or malformed part numbers will show an error message.
- The batch mode is intended for single-wavelength waveplates only; achromatic `-340` parts should be analyzed in Single File Mode.

## Customization

- Update the supported part numbers in `PART_NUMBERS` inside `ellipsometerProgram.py`.
- Adjust the plotting wavelength range in `ACHROMATIC_PLOT_RANGE_NM`.
- Change CSV parsing behavior in `compute_retardance()` if the export format differs.
