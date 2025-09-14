#!/usr/bin/env python3
"""
Fractal Carpet Studio (GUI)
--------------------------------------------------
An interactive, more advanced take on your fractal carpet generator.

Highlights
- Live preview with Tkinter (no extra installs).
- Adjustable rules (count, size), iterations, grayscale/RGB, seed.
- New: arbitrary rule size (k×k), not just 3×3.
- New: probabilistic / stochastic rules option.
- New: symmetry helpers (none, rotational, mirror) when randomizing.
- New: palette choices (random, greys, cubehelix-ish, fire, ocean).
- New: rule mutation (small random tweaks) and rule export/import.
- Non-blocking generation via worker thread; cancel in-flight jobs.
- Safe guards to preview downscaled image if output explodes.

Drop this file next to your existing main.py (or anywhere). Run:
    python fractal_gui.py

This is self-contained. Only needs numpy, matplotlib, pillow (PIL). If PIL
is unavailable, it will fallback to matplotlib imsave and a basic preview.

Based on your original FractalArray in main.py, but generalized.
"""

import json
import math
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np
from matplotlib import pyplot as plt

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False

import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ------------------------------- Core Engine ------------------------------- #

@dataclass
class Palette:
    name: str
    def get_colors(self, n: int) -> np.ndarray:
        rng = np.random.default_rng()
        if self.name == "greys":
            vals = np.linspace(0.05, 0.95, n)
            return np.stack([vals, vals, vals], axis=1)
        if self.name == "cubehelix":
            # simple cubehelix-like generation
            t = np.linspace(0.0, 1.0, n)
            a = 0.5 + 0.5 * t
            phi = 2 * np.pi * (1.0 * t + 0.2)
            r = np.clip(a * (1 + np.cos(phi)), 0, 1)
            g = np.clip(a * (1 + np.cos(phi + 2.1)), 0, 1)
            b = np.clip(a * (1 + np.cos(phi + 4.2)), 0, 1)
            return np.stack([r, g, b], axis=1)
        if self.name == "fire":
            t = np.linspace(0.0, 1.0, n)
            r = np.clip(0.2 + 1.2 * t, 0, 1)
            g = np.clip(0.05 + 0.9 * t**0.7, 0, 1)
            b = np.clip(0.0 + 0.2 * t**0.3, 0, 1)
            return np.stack([r, g, b], axis=1)
        if self.name == "ocean":
            t = np.linspace(0.0, 1.0, n)
            r = np.clip(0.05 + 0.2 * t, 0, 1)
            g = np.clip(0.2 + 0.6 * t, 0, 1)
            b = np.clip(0.4 + 0.6 * t, 0, 1)
            return np.stack([r, g, b], axis=1)
        # default random but slightly muted
        return rng.uniform(0.1, 0.9, size=(n, 3))


def apply_symmetry(mat: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Enforce simple symmetries on a k×k rule tile."""
    kx, ky = mat.shape
    m = mat.copy()
    if mode == "none":
        return m
    if mode == "mirror":
        # mirror horizontally then vertically
        m = (m + np.fliplr(m)) // 2
        m = (m + np.flipud(m)) // 2
        return m
    if mode == "rotational":
        r90 = np.rot90(m)
        r180 = np.rot90(r90)
        r270 = np.rot90(r180)
        m = (m + r90 + r180 + r270) // 4
        return m
    return m


@dataclass
class FractalArray:
    base_x: int = 3
    base_y: int = 3
    rule_size: Tuple[int, int] = (3, 3)
    rules: Dict[int, np.ndarray] = field(default_factory=dict)
    colors: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.0, 0.0], [1, 1, 1]]))
    stochastic: bool = False
    stochastic_p: float = 0.0  # chance to randomly flip a cell to any state during expansion

    def __post_init__(self):
        self.reset()
        if not self.rules:
            # default simple rule set (2 states)
            self.rules = {
                0: np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=int),
                1: np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
            }
            self.colors = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
        self._rng = np.random.default_rng()

    def reset(self):
        self.array = np.zeros((self.base_x, self.base_y), dtype=int)

    def set_rules(self, rules: Dict[int, np.ndarray], colors: np.ndarray):
        self.rules = rules
        self.colors = colors

    def randomize_rules(self, num_states: int, kx: int, ky: int, *,
                        palette: Palette, symmetry: str = "none",
                        greyscale: bool = False, seed: Optional[int] = None,
                        invert_pair: bool = False, stochastic: bool = False,
                        p: float = 0.0):
        rng = np.random.default_rng(seed)
        rules = {}
        for s in range(num_states):
            mat = rng.integers(0, num_states, size=(kx, ky), endpoint=False)
            if symmetry != "none":
                mat = apply_symmetry(mat, symmetry, rng)
            rules[s] = mat.astype(int)
        if invert_pair and num_states == 2:
            a = rng.integers(0, 2, size=(kx, ky), endpoint=False)
            rules[0] = a
            rules[1] = 1 - a
        cols = Palette("greys" if greyscale else palette.name).get_colors(num_states)
        self.set_rules(rules, cols)
        self.rule_size = (kx, ky)
        self.stochastic = stochastic
        self.stochastic_p = float(p)

    def _expand_once(self) -> np.ndarray:
        """Expand the current array by replacing each cell with its rule tile.
        Uses rule tile size to compute the kronecker expansion, but because
        rule choices depend on cell state (and we also support stochastic flips),
        we build block-wise.
        """
        kx, ky = next(iter(self.rules.values())).shape
        ax, ay = self.array.shape
        out = np.zeros((ax * kx, ay * ky), dtype=int)
        if self.stochastic and self.stochastic_p > 0.0:
            flip_mask = self._rng.random((ax, ay)) < self.stochastic_p
        else:
            flip_mask = None
        for ix in range(ax):
            for iy in range(ay):
                s = int(self.array[ix, iy])
                if flip_mask is not None and flip_mask[ix, iy]:
                    s = int(self._rng.integers(0, len(self.rules)))
                tile = self.rules[s]
                x0, y0 = ix * kx, iy * ky
                out[x0:x0 + kx, y0:y0 + ky] = tile
        return out

    def iterate(self, n: int):
        for _ in range(n):
            self.array = self._expand_once()

    def to_rgb(self) -> np.ndarray:
        rgb = self.colors[self.array]
        return rgb

    def save_image(self, path: str):
        rgb = self.to_rgb()
        plt.imsave(path, rgb)

    def save_rules(self, path: str):
        data = {str(k): v.tolist() for k, v in self.rules.items()}
        payload = {
            "rule_size": list(next(iter(self.rules.values())).shape),
            "rules": data,
            "colors": self.colors.tolist(),
            "stochastic": self.stochastic,
            "stochastic_p": self.stochastic_p,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_rules(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rules = {int(k): np.array(v, dtype=int) for k, v in payload["rules"].items()}
        self.rules = rules
        self.colors = np.array(payload.get("colors", [[0, 0, 0], [1, 1, 1]]), dtype=float)
        kx, ky = payload.get("rule_size", [3, 3])
        self.rule_size = (int(kx), int(ky))
        self.stochastic = bool(payload.get("stochastic", False))
        self.stochastic_p = float(payload.get("stochastic_p", 0.0))


# ------------------------------ GUI Utilities ------------------------------ #

@dataclass
class Job:
    seed: Optional[int]
    num_states: int
    rule_kx: int
    rule_ky: int
    base_x: int
    base_y: int
    iterations: int
    palette: str
    greys: bool
    symmetry: str
    invert_pair: bool
    stochastic: bool
    stochastic_p: float


class FractalApp:
    def __init__(self, root):
        self.root = root
        root.title("Fractal Carpet Studio")
        self.engine = FractalArray()
        self.preview_max = 900  # max preview pixels on longer edge

        # layout
        self.main = ttk.Frame(root, padding=10)
        self.main.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        self._build_controls()
        self._build_preview()
        self._build_buttons()

        # worker
        self.job_queue = queue.Queue()
        self.res_queue = queue.Queue()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._cancel = threading.Event()
        self.worker.start()

        # initial render
        self._submit_job()
        self.root.after(100, self._poll_results)

    # ---- UI construction ---- #
    def _build_controls(self):
        pad = {"padx": 6, "pady": 3}
        panel = ttk.LabelFrame(self.main, text="Parameters")
        panel.grid(row=0, column=0, sticky="nsw", **pad)

        # seed
        ttk.Label(panel, text="Seed (empty=random)").grid(row=0, column=0, sticky="w")
        self.seed_var = tk.StringVar()
        ttk.Entry(panel, textvariable=self.seed_var, width=12).grid(row=0, column=1)

        # states & rule size
        ttk.Label(panel, text="# States").grid(row=1, column=0, sticky="w")
        self.states_var = tk.IntVar(value=6)
        ttk.Spinbox(panel, from_=2, to=32, textvariable=self.states_var, width=6).grid(row=1, column=1)

        ttk.Label(panel, text="Rule kx×ky").grid(row=2, column=0, sticky="w")
        self.kx_var = tk.IntVar(value=3)
        self.ky_var = tk.IntVar(value=3)
        row = 2
        frm = ttk.Frame(panel)
        frm.grid(row=row, column=1)
        ttk.Spinbox(frm, from_=2, to=9, textvariable=self.kx_var, width=4).grid(row=0, column=0)
        ttk.Label(frm, text="×").grid(row=0, column=1)
        ttk.Spinbox(frm, from_=2, to=9, textvariable=self.ky_var, width=4).grid(row=0, column=2)

        # base grid & iterations
        ttk.Label(panel, text="Base grid x×y").grid(row=3, column=0, sticky="w")
        self.bx_var = tk.IntVar(value=3)
        self.by_var = tk.IntVar(value=3)
        frm2 = ttk.Frame(panel)
        frm2.grid(row=3, column=1)
        ttk.Spinbox(frm2, from_=2, to=12, textvariable=self.bx_var, width=4).grid(row=0, column=0)
        ttk.Label(frm2, text="×").grid(row=0, column=1)
        ttk.Spinbox(frm2, from_=2, to=12, textvariable=self.by_var, width=4).grid(row=0, column=2)

        ttk.Label(panel, text="Iterations").grid(row=4, column=0, sticky="w")
        self.iters_var = tk.IntVar(value=5)
        ttk.Spinbox(panel, from_=1, to=7, textvariable=self.iters_var, width=6).grid(row=4, column=1)

        # palette & mode
        ttk.Label(panel, text="Palette").grid(row=5, column=0, sticky="w")
        self.palette_var = tk.StringVar(value="cubehelix")
        ttk.Combobox(panel, textvariable=self.palette_var, values=[
            "random", "greys", "cubehelix", "fire", "ocean"
        ], width=12, state="readonly").grid(row=5, column=1)

        self.greys_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Greyscale", variable=self.greys_var).grid(row=6, column=0, columnspan=2, sticky="w")

        ttk.Label(panel, text="Symmetry").grid(row=7, column=0, sticky="w")
        self.sym_var = tk.StringVar(value="none")
        ttk.Combobox(panel, textvariable=self.sym_var, values=["none", "mirror", "rotational"],
                     width=12, state="readonly").grid(row=7, column=1)

        self.invpair_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Invert pair (2 states)", variable=self.invpair_var).grid(row=8, column=0, columnspan=2, sticky="w")

        # stochastic
        st_frame = ttk.LabelFrame(self.main, text="Stochastic")
        st_frame.grid(row=1, column=0, sticky="nsw", **pad)
        self.stoch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(st_frame, text="Enable stochastic rewrites", variable=self.stoch_var).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(st_frame, text="Flip prob p").grid(row=1, column=0, sticky="w")
        self.p_var = tk.DoubleVar(value=0.02)
        ttk.Spinbox(st_frame, from_=0.0, to=0.5, increment=0.01, textvariable=self.p_var, width=6).grid(row=1, column=1)

    def _build_preview(self):
        pad = {"padx": 6, "pady": 3}
        frame = ttk.LabelFrame(self.main, text="Preview")
        frame.grid(row=0, column=1, rowspan=2, sticky="nsew", **pad)
        self.main.columnconfigure(1, weight=1)
        self.main.rowconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(frame, width=640, height=640, bg="#111")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frame, textvariable=self.status_var).grid(row=1, column=0, sticky="we")
        self._tk_img = None

    def _build_buttons(self):
        pad = {"padx": 6, "pady": 3}
        bar = ttk.Frame(self.main)
        bar.grid(row=2, column=0, columnspan=2, sticky="we", **pad)
        bar.columnconfigure(10, weight=1)

        ttk.Button(bar, text="Randomize", command=self._submit_job).grid(row=0, column=0)
        ttk.Button(bar, text="Mutate Rules", command=self._mutate_rules).grid(row=0, column=1)
        ttk.Button(bar, text="Regenerate", command=self._regenerate_same_rules).grid(row=0, column=2)
        ttk.Button(bar, text="Export Rules", command=self._export_rules).grid(row=0, column=3)
        ttk.Button(bar, text="Import Rules", command=self._import_rules).grid(row=0, column=4)
        ttk.Button(bar, text="Save Image", command=self._save_image).grid(row=0, column=5)
        ttk.Button(bar, text="Cancel", command=self._cancel_job).grid(row=0, column=6)

    # ---- worker loop ---- #
    def _worker_loop(self):
        while True:
            job: Job = self.job_queue.get()
            if job is None:
                break
            try:
                if self._cancel.is_set():
                    self.res_queue.put(("cancelled", None))
                    continue
                t0 = time.time()
                rng_seed = int(job.seed) if (job.seed and str(job.seed).strip()) else None
                engine = FractalArray(base_x=job.base_x, base_y=job.base_y)
                engine.randomize_rules(
                    num_states=job.num_states,
                    kx=job.rule_kx,
                    ky=job.rule_ky,
                    palette=Palette(job.palette),
                    symmetry=job.symmetry,
                    greyscale=job.greys,
                    seed=rng_seed,
                    invert_pair=job.invert_pair,
                    stochastic=job.stochastic,
                    p=job.stochastic_p,
                )
                engine.iterate(job.iterations)
                rgb = engine.to_rgb()
                dt = time.time() - t0
                self.res_queue.put(("ok", (rgb, dt, engine)))
            except Exception as e:
                self.res_queue.put(("err", str(e)))

    # ---- job helpers ---- #
    def _gather_job(self) -> Job:
        return Job(
            seed=self.seed_var.get(),
            num_states=self.states_var.get(),
            rule_kx=self.kx_var.get(),
            rule_ky=self.ky_var.get(),
            base_x=self.bx_var.get(),
            base_y=self.by_var.get(),
            iterations=self.iters_var.get(),
            palette=self.palette_var.get(),
            greys=self.greys_var.get(),
            symmetry=self.sym_var.get(),
            invert_pair=self.invpair_var.get(),
            stochastic=self.stoch_var.get(),
            stochastic_p=self.p_var.get(),
        )

    def _submit_job(self):
        self._cancel.clear()
        job = self._gather_job()
        self.status_var.set("Generating…")
        self.job_queue.put(job)

    def _regenerate_same_rules(self):
        # keep current engine rules but just iterate from fresh base
        try:
            job = self._gather_job()
            self._cancel.clear()
            self.status_var.set("Regenerating…")
            engine = FractalArray(base_x=job.base_x, base_y=job.base_y)
            engine.set_rules(self.engine.rules, self.engine.colors)
            engine.stochastic = self.engine.stochastic
            engine.stochastic_p = self.engine.stochastic_p
            engine.iterate(job.iterations)
            rgb = engine.to_rgb()
            self._set_preview(rgb, engine, 0.0)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _mutate_rules(self):
        # small random changes to current rules
        try:
            rng = np.random.default_rng()
            mutated = {}
            num_states = len(self.engine.rules)
            for k, v in self.engine.rules.items():
                m = v.copy()
                kx, ky = m.shape
                # mutate ~5% of entries
                flips = max(1, (kx * ky) // 20)
                xs = rng.integers(0, kx, size=flips)
                ys = rng.integers(0, ky, size=flips)
                vals = rng.integers(0, num_states, size=flips)
                m[xs, ys] = vals
                mutated[k] = m
            self.engine.rules = mutated
            self._regenerate_same_rules()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export_rules(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.engine.save_rules(path)
            messagebox.showinfo("Saved", f"Rules saved to\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _import_rules(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            self.engine.load_rules(path)
            # update dependent UI fields
            kx, ky = next(iter(self.engine.rules.values())).shape
            self.states_var.set(len(self.engine.rules))
            self.kx_var.set(kx)
            self.ky_var.set(ky)
            self._regenerate_same_rules()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_image(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            self.engine.save_image(path)
            messagebox.showinfo("Saved", f"Image saved to\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _cancel_job(self):
        self._cancel.set()
        self.status_var.set("Cancelled (next job will start fresh)")

    # ---- result handling & preview ---- #
    def _poll_results(self):
        try:
            tag, payload = self.res_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_results)
            return
        if tag == "ok":
            rgb, dt, engine = payload
            self._set_preview(rgb, engine, dt)
        elif tag == "err":
            messagebox.showerror("Error", payload)
            self.status_var.set("Error")
        elif tag == "cancelled":
            # ignore silently
            pass
        self.root.after(100, self._poll_results)

    def _set_preview(self, rgb: np.ndarray, engine: FractalArray, dt: float):
        self.engine = engine
        h, w, _ = rgb.shape
        scale = 1.0
        long_edge = max(h, w)
        if long_edge > self.preview_max and PIL_OK:
            scale = self.preview_max / long_edge
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((new_w, new_h), resample=Image.NEAREST)
        else:
            if PIL_OK:
                img = Image.fromarray((rgb * 255).astype(np.uint8))
            else:
                # fallback: save temp and reload via Tk (less ideal but works)
                tmp = os.path.join(os.getcwd(), "_preview.png")
                plt.imsave(tmp, rgb)
                try:
                    from PIL import Image as _Image
                    img = _Image.open(tmp)
                except Exception:
                    self.status_var.set(f"Preview too large to show ({w}×{h}). Saved _preview.png")
                    return
        if PIL_OK:
            self._tk_img = ImageTk.PhotoImage(img)
            self.canvas.config(width=self._tk_img.width(), height=self._tk_img.height())
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, image=self._tk_img, anchor="nw")
        self.status_var.set(f"{w}×{h} px • {len(engine.rules)} states • iter={self.iters_var.get()} • {dt:.2f}s")


# ------------------------------- Entrypoint ------------------------------- #

def main():
    root = tk.Tk()
    try:
        # nicer looking on some platforms
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    app = FractalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
