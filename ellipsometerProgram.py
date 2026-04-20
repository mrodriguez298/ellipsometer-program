'''
Desktop GUI for Thorlabs ellipsometer CSV exports.

The operator selects a waveplate part number and a CSV file, the app runs a Lu-Chipman 
decomposition on every row (one 4x4 Mueller matrix per wavelength) and reports:
  - Single-wavelength waveplates (WPH*/WPQ*): retardance at the design
    wavelength plus a PASS/FAIL verdict against the Thorlabs tolerance.
  - Broadband achromatics ending in -340 (AHWP*-340 / AQWP*-340):
    a retardance-vs-wavelength plot from 260 nm to 410 nm.

Created by Miguel Rodriguez (Seasonal Engineer) Spring 2026.
'''

from __future__ import annotations
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)

PART_NUMBERS: list[str] = [
    "WPQSM05-266", "WPQSM05-308", "WPQSM05-325",
    "WPQSM05-343", "WPQSM05-355", "WPQSM05-370", "WPQSM05-400",
    "WPQ05M-266", "WPQ05M-308", "WPQ05M-325",
    "WPQ05M-343", "WPQ05M-355", "WPQ05M-370", "WPQ05M-400",
    "WPQ10M-266", "WPQ10M-308", "WPQ10M-325",
    "WPQ10M-343", "WPQ10M-355", "WPQ10M-370", "WPQ10M-400",
    "WPHSM05-266", "WPHSM05-308", "WPHSM05-325",
    "WPHSM05-343", "WPHSM05-355", "WPHSM05-370", "WPHSM05-400",
    "WPH05M-266", "WPH05M-308", "WPH05M-325",
    "WPH05M-343", "WPH05M-355", "WPH05M-370", "WPH05M-400",
    "WPH10M-266", "WPH10M-308", "WPH10M-325",
    "WPH10M-343", "WPH10M-355", "WPH10M-370", "WPH10M-400",
    "AQWP05M-340", "AQWP10M-340",
    "AHWP05M-340", "AHWP10M-340",
]
ACHROMATIC_PLOT_RANGE_NM = (260, 410)

def parse_part_number(pn: str) -> dict:
    '''
    Classify a Thorlabs waveplate part number.

    Returns dict with keys:
        plate_type: 'H' or 'Q'
        achromatic: True for AHWP*/AQWP*, False for WPH*/WPQ*
        wavelength_nm: design wavelength parsed from the suffix
    '''

    s = pn.strip().upper()
    if s.startswith("AHWP"):
        plate_type, achromatic = "H", True
    elif s.startswith("AQWP"):
        plate_type, achromatic = "Q", True
    elif s.startswith("WPH"):
        plate_type, achromatic = "H", False
    elif s.startswith("WPQ"):
        plate_type, achromatic = "Q", False
    else:
        raise ValueError(f"Unrecognized part number: {pn!r}")

    m = re.search(r"-(\d+)\s*$", s)
    if not m:
        raise ValueError(f"Could not parse design wavelength from: {pn!r}")
    return {
        "plate_type": plate_type,
        "achromatic": achromatic,
        "wavelength_nm": int(m.group(1)),
    }

def tolerance_for(plate_type: str) -> tuple[float, float, float]:
    '''
    Returns acceptance bounds in waves for an H or Q plate.
    '''
    if plate_type == "H":
        return 0.496, 0.504, 0.500
    return 0.246, 0.254, 0.250

def retardance_at_design_wavelength(
    wavelengths: np.ndarray,
    retardance: np.ndarray,
    target_nm: int,
) -> tuple[float, float, float]:
    '''
    Interpolate retardance to the design wavelength. Returns (retardance_at_target, data_min_nm, data_max_nm).
    '''
    order = np.argsort(wavelengths)
    wl_sorted = wavelengths[order]
    ret_sorted = retardance[order]
    ret_at = float(np.interp(target_nm, wl_sorted, ret_sorted))
    return ret_at, float(wl_sorted.min()), float(wl_sorted.max())

def compute_retardance(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Run Lu-Chipman decomposition on an ellipsometer CSV export.

    Returns (wavelengths_nm, retardance_waves) where retardance is expressed in units of waves (0.5 = half-wave, 0.25 = quarter-wave).
    '''
    df = pd.read_csv(csv_path, skiprows=1)
    df = df.drop(df.iloc[:, 17:], axis=1)
    MM_array = df.to_numpy()
    nolam = np.delete(MM_array, 0, 1)

    reshaped_arrays: list[np.ndarray] = []
    for row in np.vsplit(nolam, nolam.shape[0]):
        if row.size == 16:
            reshaped_arrays.append(row.reshape((4, 4)))

    wavelengths: list[float] = []
    retardances: list[float] = []

    for idx, array in enumerate(reshaped_arrays):
        T_u = array[0][0]
        m = array[1:4, 1:4]

        d_vecti = (1.0 / T_u) * np.delete(array[0], 0)
        d_vect = np.insert(d_vecti, 0, 0)
        D = np.sqrt(d_vect[1] ** 2 + d_vect[2] ** 2 + d_vect[3] ** 2)
        D_hat = np.reshape(d_vect / D, (1, 4))
        D_hat_T = np.transpose(D_hat)

        P_vecti = (1.0 / T_u) * np.delete(array[:, 0], 0)
        P_vect = np.insert(P_vecti, 0, 0)

        M_D = np.identity(4)
        M_D[0] = M_D[0] + d_vect
        M_D[:, 0] = M_D[:, 0] + P_vect
        m_D = (
            np.sqrt(1 - D ** 2) * np.identity(3)
            + (1 - np.sqrt(1 - D ** 2))
            * np.delete(D_hat, 0)
            * np.delete(D_hat_T, 0)
        )
        P_delta = (P_vecti - np.dot(m, d_vecti)) / (1 - D ** 2)
        P_delta2 = np.insert(P_delta, 0, 0)

        M_D[1:4, 1:4] = m_D
        M_D = T_u * M_D
        M_prime = np.dot(array, np.linalg.inv(M_D))

        m_prime = M_prime[1:4, 1:4]
        m_prime_T = np.transpose(m_prime)
        m_p = np.dot(m_prime, m_prime_T)

        eigenvalues = np.linalg.eigvals(m_p)

        m_delta = np.dot(
            np.linalg.inv(
                m_p
                + (
                    np.sqrt(eigenvalues[0] * eigenvalues[1])
                    + np.sqrt(eigenvalues[1] * eigenvalues[2])
                    + np.sqrt(eigenvalues[2] * eigenvalues[0])
                )
                * np.identity(3)
            ),
            np.dot(
                np.sqrt(eigenvalues[0])
                + np.sqrt(eigenvalues[1])
                + np.sqrt(eigenvalues[2]),
                m_p,
            )
            + np.sqrt(eigenvalues[0] * eigenvalues[1] * eigenvalues[2])
            * np.identity(3),
        )

        if np.linalg.det(m_prime) < 0:
            m_delta = -m_delta

        M_delta = np.identity(4)
        M_delta[1:4, 1:4] = m_delta
        M_delta[:, 0] = M_delta[:, 0] + P_delta2

        if np.linalg.det(M_delta) == 0 or np.linalg.det(M_prime) == 0:
            M_R = np.identity(4)
        else:
            M_R = np.dot(np.linalg.inv(M_delta), M_prime)

        trace_arg = np.clip(np.real(np.trace(M_R) / 2 - 1), -1.0, 1.0)
        ret = np.arccos(trace_arg) / (2 * np.pi)

        wavelengths.append(float(MM_array[idx, 0]))
        retardances.append(float(np.real(ret)))

    return np.array(wavelengths), np.array(retardances)

class SingleFileTab(ttk.Frame):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent, padding=12)

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Part Number:").grid(row=0, column=0, sticky="w")
        self.part_var = tk.StringVar()
        ttk.Combobox(
            controls,
            textvariable=self.part_var,
            values=PART_NUMBERS,
            width=24,
        ).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(controls, text="CSV File:").grid(row=0, column=2, sticky="w")
        self.file_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.file_var, width=48).grid(
            row=0, column=3, sticky="we", padx=(6, 6)
        )
        ttk.Button(controls, text="Browse…", command=self._browse).grid(
            row=0, column=4, padx=(0, 6)
        )
        ttk.Button(controls, text="Run Analysis", command=self._run).grid(
            row=0, column=5
        )
        controls.columnconfigure(3, weight=1)

        self.result_frame = ttk.Frame(self, padding=(0, 12, 0, 0))
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        self._placeholder("Select a part number and CSV file, then click Run Analysis.")

    def _clear(self) -> None:
        for w in self.result_frame.winfo_children():
            w.destroy()

    def _placeholder(self, text: str) -> None:
        self._clear()
        tk.Label(
            self.result_frame, text=text, font=("Segoe UI", 11), fg="#555"
        ).pack(expand=True)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select ellipsometer CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.file_var.set(path)

    def _run(self) -> None:
        pn = self.part_var.get().strip()
        path = self.file_var.get().strip()
        if not pn:
            messagebox.showerror("Missing input", "Select a part number.")
            return
        if not path or not os.path.isfile(path):
            messagebox.showerror("Missing input", "Select a valid CSV file.")
            return

        try:
            info = parse_part_number(pn)
        except ValueError as exc:
            messagebox.showerror("Invalid part number", str(exc))
            return

        try:
            wavelengths, retardance = compute_retardance(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Analysis failed", f"{type(exc).__name__}: {exc}")
            return

        self._clear()
        if info["achromatic"] and info["wavelength_nm"] == 340:
            self._show_plot(wavelengths, retardance, pn)
        else:
            self._show_single_wavelength(wavelengths, retardance, pn, info)

    def _show_single_wavelength(
        self,
        wavelengths: np.ndarray,
        retardance: np.ndarray,
        pn: str,
        info: dict,
    ) -> None:
        target = info["wavelength_nm"]
        ret_at, wl_min, wl_max = retardance_at_design_wavelength(
            wavelengths, retardance, target
        )
        if target < wl_min or target > wl_max:
            messagebox.showwarning(
                "Out of range",
                f"Design wavelength {target} nm is outside the data range "
                f"({wl_min:.1f}–{wl_max:.1f} nm). "
                "Extrapolating to the nearest measured point.",
            )

        lo, hi, nominal = tolerance_for(info["plate_type"])
        passed = lo <= ret_at <= hi

        tk.Label(
            self.result_frame,
            text="PASS" if passed else "FAIL",
            font=("Segoe UI", 48, "bold"),
            fg="white",
            bg=("#2e7d32" if passed else "#9c0f0f"),
            pady=24,
        ).pack(fill=tk.X)

        details = (
            f"Part number:        {pn}\n"
            f"Design wavelength:  {target} nm\n"
            f"Measured retardance: {ret_at:.4f} waves  (nominal {nominal:.3f})\n"
            f"Acceptance window:  {lo:.3f} – {hi:.3f} waves"
        )
        tk.Label(
            self.result_frame,
            text=details,
            font=("Consolas", 13),
            justify="left",
            anchor="w",
            padx=20,
            pady=20,
        ).pack(fill=tk.BOTH, expand=True)

    def _show_plot(
        self,
        wavelengths: np.ndarray,
        retardance: np.ndarray,
        pn: str,
    ) -> None:
        lo_nm, hi_nm = ACHROMATIC_PLOT_RANGE_NM
        order = np.argsort(wavelengths)
        wl_sorted = wavelengths[order]
        ret_sorted = retardance[order]
        mask = (wl_sorted >= lo_nm) & (wl_sorted <= hi_nm)

        fig = Figure(figsize=(7.2, 4.6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(wl_sorted[mask], ret_sorted[mask], color="#1565c0", linewidth=1.8)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Retardance (waves)")
        ax.set_title(f"{pn} — retardance vs. wavelength")
        ax.set_xlim(lo_nm, hi_nm)
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, self.result_frame).update()

class BatchTab(ttk.Frame):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent, padding=12)

        self._paths: list[str] = []
        self._last_rows: list[dict] = []

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X)

        ttk.Label(controls, text="Part Number:").grid(row=0, column=0, sticky="w")
        self.part_var = tk.StringVar()
        ttk.Combobox(
            controls,
            textvariable=self.part_var,
            values=PART_NUMBERS,
            width=24,
        ).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Button(controls, text="Select Files…", command=self._select_files).grid(
            row=0, column=2, padx=(0, 6)
        )
        self.count_var = tk.StringVar(value="no files selected")
        ttk.Label(controls, textvariable=self.count_var).grid(
            row=0, column=3, sticky="w", padx=(0, 18)
        )
        ttk.Button(controls, text="Run Batch", command=self._run).grid(
            row=0, column=4, padx=(0, 6)
        )
        ttk.Button(controls, text="Export CSV…", command=self._export).grid(
            row=0, column=5
        )
        controls.columnconfigure(3, weight=1)

        table_frame = ttk.Frame(self)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 6))

        cols = ("file", "part", "wavelength", "retardance", "window", "verdict")
        self.tree = ttk.Treeview(
            table_frame, columns=cols, show="headings", height=15
        )
        headings = {
            "file": ("File", 260),
            "part": ("Part #", 110),
            "wavelength": ("λ (nm)", 70),
            "retardance": ("Retardance", 100),
            "window": ("Window", 120),
            "verdict": ("Verdict", 80),
        }
        for c, (label, w) in headings.items():
            self.tree.heading(c, text=label)
            self.tree.column(c, width=w, anchor="w" if c in ("file", "part") else "center")
        self.tree.tag_configure("pass", background="#e8f5e9", foreground="#1b5e20")
        self.tree.tag_configure("fail", background="#ffebee", foreground="#b71c1c")
        self.tree.tag_configure("error", background="#fff3e0", foreground="#e65100")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.summary_var = tk.StringVar(value="Select a part number and files, then click Run Batch.")
        ttk.Label(self, textvariable=self.summary_var, font=("Segoe UI", 10)).pack(
            fill=tk.X
        )

    def _select_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select ellipsometer CSVs",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if paths:
            self._paths = list(paths)
            self.count_var.set(f"{len(self._paths)} file(s) selected")

    def _clear_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _run(self) -> None:
        pn = self.part_var.get().strip()
        if not pn:
            messagebox.showerror("Missing input", "Select a part number.")
            return
        if not self._paths:
            messagebox.showerror("Missing input", "Select one or more CSV files.")
            return

        try:
            info = parse_part_number(pn)
        except ValueError as exc:
            messagebox.showerror("Invalid part number", str(exc))
            return

        if info["achromatic"] and info["wavelength_nm"] == 340:
            messagebox.showinfo(
                "Use Single File mode",
                "Achromatic -340 parts are characterized by a retardance-vs-wavelength "
                "plot, not a single PASS/FAIL. Switch to the Single File tab to view "
                "individual plots.",
            )
            return

        target = info["wavelength_nm"]
        lo, hi = tolerance_for(info["plate_type"])[:2]
        window = f"{lo:.3f}–{hi:.3f}"

        self._clear_table()
        self._last_rows = []
        n_pass = n_fail = n_err = 0

        for path in self._paths:
            name = os.path.basename(path)
            try:
                wavelengths, retardance = compute_retardance(path)
                ret_at, _, _ = retardance_at_design_wavelength(
                    wavelengths, retardance, target
                )
                passed = lo <= ret_at <= hi
                verdict = "PASS" if passed else "FAIL"
                tag = "pass" if passed else "fail"
                ret_str = f"{ret_at:.4f}"
                if passed:
                    n_pass += 1
                else:
                    n_fail += 1
                row = {
                    "file": name,
                    "part": pn,
                    "wavelength": target,
                    "retardance": ret_str,
                    "window": window,
                    "verdict": verdict,
                }
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                tag = "error"
                row = {
                    "file": name,
                    "part": pn,
                    "wavelength": target,
                    "retardance": "—",
                    "window": window,
                    "verdict": f"ERROR: {type(exc).__name__}",
                }

            self._last_rows.append(row)
            self.tree.insert(
                "",
                "end",
                values=(
                    row["file"],
                    row["part"],
                    row["wavelength"],
                    row["retardance"],
                    row["window"],
                    row["verdict"],
                ),
                tags=(tag,),
            )

        total = len(self._paths)
        self.summary_var.set(
            f"{total} file(s) processed — PASS: {n_pass}   FAIL: {n_fail}   ERROR: {n_err}"
        )

    def _export(self) -> None:
        if not self._last_rows:
            messagebox.showinfo("Nothing to export", "Run a batch first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save batch report",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        df = pd.DataFrame(self._last_rows)
        df = df.rename(
            columns={
                "file": "File",
                "part": "Part #",
                "wavelength": "Wavelength (nm)",
                "retardance": "Retardance (waves)",
                "window": "Acceptance window",
                "verdict": "Verdict",
            }
        )
        df.to_csv(path, index=False)
        messagebox.showinfo("Exported", f"Report saved to:\n{path}")

class EllipsometerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Ellipsometer Analyzer")
        root.geometry("980x680")
        root.minsize(820, 560)

        notebook = ttk.Notebook(root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        notebook.add(SingleFileTab(notebook), text="Single File Mode")
        notebook.add(BatchTab(notebook), text="Batch Mode")

def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    EllipsometerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
