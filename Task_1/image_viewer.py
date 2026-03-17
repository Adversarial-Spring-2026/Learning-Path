# Using Claude

# =============================================================================
#  IMAGE COMPARISON VIEWER  —  samples vs labels
#
#  PURPOSE:
#    Side-by-side viewer that pairs images from two folders ("samples" and
#    "labels") by their shared base name, lets you iterate through them, and
#    flags any mismatches (a file present in one folder but not the other).
#
#  HOW TO USE IN JUPYTER:
#    1. Place this file in the same folder as your "samples/" and "labels/" dirs.
#    2. In a notebook cell, run:
#           from image_viewer import ImageViewer
#           viewer = ImageViewer()
#           viewer.show()
#    3. Use the buttons to navigate, or press ← → on your keyboard.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# Each library has a specific job:
# -----------------------------------------------------------------------------

import os          # os = "operating system" — lets us work with folders and file paths
import re          # re = "regular expressions" — lets us search/match text patterns in strings
from pathlib import Path  # Path is a modern, cleaner way to handle file paths than plain strings

import matplotlib.pyplot as plt          # matplotlib is the main plotting library; pyplot is its drawing interface
import matplotlib.image as mpimg         # mpimg gives us the imread() function to load image files into arrays
from matplotlib.widgets import Button    # Button lets us add clickable buttons to a matplotlib figure
import ipywidgets as widgets             # ipywidgets provides interactive widgets (sliders, dropdowns, etc.) for Jupyter
from IPython.display import display      # display() renders widgets and figures inside a Jupyter notebook cell


# =============================================================================
#  SECTION 1 — FILE COLLECTION
#  Goal: find all image files inside a given folder.
# =============================================================================

def collect_files(folder: str) -> list[Path]:
    """
    Scans a folder and returns a sorted list of Path objects for every image file found.

    Parameters
    ----------
    folder : str
        The name (or path) of the folder to scan, e.g. "samples"

    Returns
    -------
    list[Path]
        A sorted list of Path objects, one per image file found.
        Example: [PosixPath('samples/foo.SAFE.img_001.tiff'), ...]

    What is a Path object?
        Instead of treating file paths as plain strings like "samples/photo.png",
        Path objects understand the file system. They let you do things like:
            p = Path("samples/photo.png")
            p.name      → "photo.png"
            p.stem      → "photo"
            p.suffix    → ".png"
            p.parent    → Path("samples")
        This makes file handling much more readable and reliable across operating systems.
    """

    # Path(folder) converts the string "samples" into a Path object
    folder_path = Path(folder)

    # Check if the folder actually exists before trying to scan it
    if not folder_path.is_dir():
        raise FileNotFoundError(
            f'Folder "{folder}" not found.\n'
            f'Current working directory is: {Path.cwd()}'
        )

    # List comprehension: builds a list by looping with a condition inline.
    # folder_path.iterdir() yields every item (file or subfolder) inside the folder.
    # We keep only items where:
    #   f.is_file()                 → it's a file (not a subfolder)
    #   f.suffix.lower() == ".tiff" → its extension matches our list
    # .lower() normalizes ".PNG" and ".png" to the same thing.
    files = [
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() == ".tiff"
    ]

    # sorted() returns a new list in alphabetical order.
    # Sorting ensures consistent ordering across runs and operating systems.
    return sorted(files)


# =============================================================================
#  SECTION 2 — FILENAME PARSING
#  Goal: extract the "base name" and numeric index from filenames like
#        "SAFE.somename.img_003.png"  →  base="SAFE.somename", index=3
#        "SAFE.somename.ndv_003.png"  →  base="SAFE.somename", index=3
#  This is how we know which sample pairs with which label.
# =============================================================================

# Compile the regex pattern once here (faster than compiling inside a loop).
#
# What is a regular expression (regex)?
#   A regex is a mini-language for describing text patterns.
#   re.compile() turns a pattern string into a reusable "pattern object".
#
# Breaking down the pattern:  r"^(.*?)\.(img|ndv)_(\d+)(\.[^.]+)$"
#
#   ^           → start of string (match must begin here)
#   (.*?)       → GROUP 1: capture any characters, as few as possible ("lazy")
#                 This will become our "base name" (e.g., "SAFE.somename")
#   \.          → a literal dot  (backslash escapes the dot, since bare . means "any char")
#   (img|ndv)   → GROUP 2: capture either the word "img" or "ndv"
#   _           → a literal underscore
#   (\d+)       → GROUP 3: capture one or more digits  (\d = any digit 0–9)
#                 This will become our numeric index (e.g., "003")
#   (\.[^.]+)   → GROUP 4: capture the file extension (dot + one or more non-dot chars)
#   $           → end of string
#
#   re.IGNORECASE makes it match "IMG", "img", "Img", etc.

_FILENAME_PATTERN = re.compile(
    # Match filenames like:
    #   BASE.SAFE_ndvi_0.tiff   (dot, optional intervening token+underscore, tag_index)
    #   BASE.somename.img_003.tiff
    # GROUPS: 1=base-before-dot, 2=tag (img|ndvi), 3=index digits, 4=extension
    r"^(.*?)\.(?:.*?_)?(img|ndvi)_(\d+)(\.[^.]+)$",
    re.IGNORECASE
)


def parse_name(path: Path) -> dict:
    """
    Parses one image file's Path into its components.

    Parameters
    ----------
    path : Path
        The full path to the file, e.g. Path("samples/SAFE.foo.img_003.png")

    Returns
    -------
    dict with keys:
        "path"      → the original Path object
        "base"      → the shared base name used for pairing, e.g. "SAFE.foo"
        "tag"       → "img", "ndv", or "?" (if the pattern didn't match)
        "index"     → integer index, e.g. 3
        "key"       → the pairing key = base + "_" + zero-padded index
                      e.g.  "SAFE.foo_000003"
    """

    # path.name gives just the filename without the folder part
    # e.g.  Path("samples/SAFE.foo.img_003.png").name  →  "SAFE.foo.img_003.png"
    filename = path.name

    # Try to match the filename against our compiled regex pattern.
    # .match() checks for a match starting at the beginning of the string.
    # If it matches, m is a "match object"; if not, m is None.
    m = _FILENAME_PATTERN.match(filename)

    if m:
        # m.group(1), m.group(2), m.group(3) correspond to the three capture groups
        base  = m.group(1)           # e.g. "SAFE.somename"
        tag   = m.group(2).lower()   # e.g. "img" or "ndv"  (.lower() = lowercase)
        index = int(m.group(3))      # int() converts "003" → 3

        # The pairing key ties a sample to its label regardless of img/ndv.
        # We zero-pad the index to 6 digits so sorting works correctly:
        # "000003" sorts before "000010", while plain "3" vs "10" would not.
        # f-strings (f"...") let you embed variables directly: f"{index:06d}" formats
        # the integer `index` as a zero-padded 6-digit decimal string.
        key = f"{base}_{index:06d}"
    else:
        # Fallback for files that don't match our pattern.
        # We use the full filename as the base so they still appear in the viewer.
        base  = filename
        tag   = "?"
        index = 0
        key   = f"{base}_{0:06d}"

    return {"path": path, "base": base, "tag": tag, "index": index, "key": key}


# =============================================================================
#  SECTION 3 — PAIRING
#  Goal: combine the parsed info from both folders into a unified list of pairs,
#        marking any entries where one side is missing.
# =============================================================================

def build_pairs(samples_dir: str = "samples", labels_dir: str = "labels") -> list[dict]:
    """
    Collects files from both folders, parses their names, and pairs them by key.

    Parameters
    ----------
    samples_dir : str
        Folder containing the sample images (default: "samples")
    labels_dir : str
        Folder containing the label images (default: "labels")

    Returns
    -------
    list of dicts, each representing one pair:
        {
            "key":      str   — the shared pairing key
            "sample":   Path or None  — path to the sample image (None if missing)
            "label":    Path or None  — path to the label image  (None if missing)
            "mismatch": bool  — True if one side is missing
        }
    """

    # Collect and parse all files from each folder
    # The result is a list of dicts (one per file), as returned by parse_name()
    sample_infos = [parse_name(f) for f in collect_files(samples_dir)]
    label_infos  = [parse_name(f) for f in collect_files(labels_dir)]

    # Build dictionaries (hash maps) keyed by the pairing key.
    # Store the full parsed info dict for each key so we can access tag/index later.
    # This creates fast O(1) lookups: sample_map["SAFE.foo_000003"] → {"path": Path(...), "base": ..., "tag": ..., ...}
    sample_map = {info["key"]: info for info in sample_infos}
    label_map  = {info["key"]: info for info in label_infos}

    # Combine all unique keys from both folders.
    # set() removes duplicates; the | operator is "union" (all keys from either set).
    # sorted() converts the set back to a sorted list.
    all_keys = sorted(set(sample_map.keys()) | set(label_map.keys()))

    if not all_keys:
        raise ValueError("No image pairs found. Check that your folders contain images.")

    # Build the final pair list.
    # dict.get(key) returns the value if the key exists, or None if it doesn't.
    # This is safer than dict[key] which would raise a KeyError for missing entries.
    pairs = []
    for key in all_keys:
        sample_info = sample_map.get(key)   # None if this key has no sample
        label_info  = label_map.get(key)    # None if this key has no label

        sample_path = sample_info["path"] if sample_info is not None else None
        label_path  = label_info["path"]  if label_info is not None  else None

        mismatch    = (sample_path is None) or (label_path is None)

        pairs.append({
            "key":        key,
            "sample":     sample_path,
            "label":      label_path,
            "sample_info": sample_info,
            "label_info":  label_info,
            "mismatch":   mismatch,
        })

    return pairs


# =============================================================================
#  SECTION 4 — THE VIEWER CLASS
#  Goal: wrap all the display logic into a reusable, self-contained object.
#
#  What is a class?
#    A class is a blueprint for creating objects. An object bundles together
#    data (attributes) and functions that operate on that data (methods).
#    Here, ImageViewer is our class. When you write  viewer = ImageViewer(),
#    you create one instance (object) of it.
#
#    self  is how a method refers to its own object.
#    self.current  means "the 'current' variable belonging to this specific instance".
# =============================================================================

class ImageViewer:
    """
    Interactive side-by-side image viewer for Jupyter Notebook.

    Usage:
        viewer = ImageViewer()          # uses "samples/" and "labels/" by default
        viewer.show()                   # renders the widget in the notebook

    Optional:
        viewer = ImageViewer(samples_dir="my_samples", labels_dir="my_labels")
    """

    def __init__(self, samples_dir: str = "samples", labels_dir: str = "labels"):
        """
        __init__ is the constructor — it runs automatically when you do ImageViewer().
        It sets up all the data before anything is drawn on screen.
        """

        # Resolve folder paths intelligently:
        # 1) If the given path exists as-is (absolute or relative to cwd), use it.
        # 2) Otherwise, try resolving relative to the project root (two levels up from this file):
        #      <repo-root>/samples  (this matches your workspace layout where "samples" is outside Task_1)
        base_dir = Path(__file__).resolve().parent.parent

        def resolve_folder(folder_name: str) -> str:
            p = Path(folder_name)
            if p.is_dir():
                return str(p)
            alt = base_dir / folder_name
            if alt.is_dir():
                return str(alt)
            # return the original string and let build_pairs raise a clear error if missing
            return folder_name

        samples_path = resolve_folder(samples_dir)
        labels_path  = resolve_folder(labels_dir)

        # Build the list of image pairs (may raise errors if folders are missing)
        self.pairs   = build_pairs(samples_path, labels_path)

        # Total number of pairs
        self.n       = len(self.pairs)

        # Index of the currently displayed pair (0-based in Python, unlike MATLAB's 1-based)
        self.current = 0

        # Pre-compute a list of indices where mismatch=True, for the "jump" buttons
        # enumerate(self.pairs) yields (index, pair_dict) tuples as we loop
        self.mismatch_indices = [i for i, p in enumerate(self.pairs) if p["mismatch"]]

        # Print a summary to the console/output area
        self._print_summary()

    # -------------------------------------------------------------------------
    #  _print_summary  —  report mismatches in text before the viewer opens
    #  Methods starting with _ are "private by convention" — internal use only.
    # -------------------------------------------------------------------------

    def _print_summary(self):
        """Prints a mismatch report to the notebook output area."""

        n_mis = len(self.mismatch_indices)

        if n_mis == 0:
            print(f"✓  All {self.n} pairs matched perfectly.\n")
        else:
            print(f"⚠  Found {n_mis} mismatch(es) out of {self.n} pairs:\n")
            for i in self.mismatch_indices:
                p = self.pairs[i]
                if p["sample"] is None:
                    # f-string: embeds i+1 (1-based for readability) and the key
                    print(f"  [{i+1:3d}]  MISSING in samples: {p['key']}")
                elif p["label"] is None:
                    print(f"  [{i+1:3d}]  MISSING in labels:  {p['key']}")
            print()

    # -------------------------------------------------------------------------
    #  show  —  builds and displays the full interactive widget
    # -------------------------------------------------------------------------

    def show(self):
        """
        Builds the matplotlib figure, attaches ipywidgets buttons,
        and displays everything in the Jupyter notebook cell.
        """

        # -- FIGURE & AXES ----------------------------------------------------
        # plt.figure() creates a new blank drawing canvas.
        # figsize=(width, height) is in inches; facecolor sets the background color.
        # Colors here are given as CSS hex strings like "#1e1e24".
        fig = plt.figure(figsize=(14, 7), facecolor="#1e1e24")

        # fig.add_subplot(rows, cols, position) divides the figure into a grid
        # and returns an "Axes" object — the actual area where images are drawn.
        # (1, 2, 1) = 1 row, 2 columns, first position (left)
        ax_sample = fig.add_subplot(1, 2, 1)

        # (1, 2, 2) = 1 row, 2 columns, second position (right)
        ax_label  = fig.add_subplot(1, 2, 2)

        # Style both axes: dark background, hide tick marks (they're meaningless for images)
        for ax in (ax_sample, ax_label):
            ax.set_facecolor("#0d0d10")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # fig.text() places text at a position in the figure (0.0–1.0 = left–right / bottom–top)
        # We'll update this text's content every time we navigate.
        info_text = fig.text(
            0.5, 0.97,            # x=center, y=near top
            "",                   # placeholder; filled in by _render()
            ha="center",          # horizontal alignment
            va="top",             # vertical alignment
            fontsize=11,
            fontfamily="monospace",
            color="#dddddd"
        )

        # Store references to all the drawing objects so _render() can update them.
        # These become instance attributes (self.xxx) so all methods can access them.
        self.fig       = fig
        self.ax_sample = ax_sample
        self.ax_label  = ax_label
        self.info_text = info_text

        # Draw the first pair immediately
        self._render()

        # plt.tight_layout() adjusts spacing so titles/labels don't overlap
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # leave room at bottom for buttons

        # Detect whether we're running inside a Jupyter notebook (ZMQInteractiveShell)
        in_notebook = False
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip is not None:
                shell_name = ip.__class__.__name__
                in_notebook = shell_name == "ZMQInteractiveShell"
        except Exception:
            in_notebook = False

        if in_notebook:
            # In notebooks: show the figure (non-blocking) and render ipywidgets below it.
            plt.show()

            # -- IPYWIDGETS BUTTONS -----------------------------------------------
            def btn(label, color="#2e2e36", font_color="white"):
                b = widgets.Button(
                    description=label,
                    style={"button_color": color, "font_weight": "bold"},
                    layout=widgets.Layout(width="140px", height="36px")
                )
                try:
                    b.style.text_color = font_color
                except Exception:
                    pass
                return b

            b_first    = btn("⏮  First")
            b_prev     = btn("◀  Prev")
            b_next     = btn("Next  ▶")
            b_last     = btn("Last  ⏭")
            b_prev_mis = btn("⚠ Prev Mismatch", color="#5a2a10", font_color="#ffaa44")
            b_next_mis = btn("Next Mismatch ⚠", color="#5a2a10", font_color="#ffaa44")

            b_first.on_click(lambda _: self._go(0))
            b_prev.on_click( lambda _: self._go(self.current - 1))
            b_next.on_click( lambda _: self._go(self.current + 1))
            b_last.on_click( lambda _: self._go(self.n - 1))

            b_prev_mis.on_click(lambda _: self._jump_mismatch(-1))
            b_next_mis.on_click(lambda _: self._jump_mismatch(+1))

            nav_row     = widgets.HBox([b_first, b_prev, b_next, b_last])
            mismatch_row = widgets.HBox([b_prev_mis, b_next_mis])
            controls = widgets.VBox([nav_row, mismatch_row])
            display(controls)
        else:
            # Not running in a notebook — create matplotlib buttons inside the figure
            # so they work in standalone GUI windows (and before plt.show()).
            ax_first = fig.add_axes([0.05, 0.02, 0.11, 0.05])
            ax_prev  = fig.add_axes([0.17, 0.02, 0.11, 0.05])
            ax_next  = fig.add_axes([0.29, 0.02, 0.11, 0.05])
            ax_last  = fig.add_axes([0.41, 0.02, 0.11, 0.05])
            ax_prev_mis = fig.add_axes([0.53, 0.02, 0.18, 0.05])
            ax_next_mis = fig.add_axes([0.73, 0.02, 0.18, 0.05])

            b_first = Button(ax_first, '⏮  First')
            b_prev  = Button(ax_prev,  '◀  Prev')
            b_next  = Button(ax_next,  'Next  ▶')
            b_last  = Button(ax_last,  'Last  ⏭')
            b_prev_mis = Button(ax_prev_mis, '⚠ Prev Mismatch')
            b_next_mis = Button(ax_next_mis, 'Next Mismatch ⚠')

            b_first.on_clicked(lambda event: self._go(0))
            b_prev.on_clicked( lambda event: self._go(self.current - 1))
            b_next.on_clicked( lambda event: self._go(self.current + 1))
            b_last.on_clicked( lambda event: self._go(self.n - 1))
            b_prev_mis.on_clicked(lambda event: self._jump_mismatch(-1))
            b_next_mis.on_clicked(lambda event: self._jump_mismatch(+1))

            # Keyboard navigation (left/right/home/end)
            def _on_key(event):
                key = getattr(event, 'key', None)
                if key in ('left',):
                    self._go(self.current - 1)
                elif key in ('right',):
                    self._go(self.current + 1)
                elif key in ('home',):
                    self._go(0)
                elif key in ('end',):
                    self._go(self.n - 1)

            fig.canvas.mpl_connect('key_press_event', _on_key)

            plt.show()

    # -------------------------------------------------------------------------
    #  _render  —  redraws the figure for the current pair index
    # -------------------------------------------------------------------------

    def _render(self):
        """
        Clears and redraws both image panels for self.pairs[self.current].
        Called every time navigation changes the current index.
        """

        # Retrieve the current pair dictionary
        pair = self.pairs[self.current]

        # -- INFO BAR ---------------------------------------------------------
        tag = "  ⚠ MISMATCH" if pair["mismatch"] else ""
        info = f"[ {self.current + 1} / {self.n} ]   Key: {pair['key']}{tag}"
        self.info_text.set_text(info)
        # Change text color: orange for mismatches, light grey for normal
        self.info_text.set_color("#ff9933" if pair["mismatch"] else "#cccccc")

        # -- SAMPLE PANEL (LEFT) ----------------------------------------------
        self.ax_sample.cla()   # cla() = "clear axes" — removes previous image

        if pair["sample"] is not None:
            # mpimg.imread() loads an image file into a NumPy array
            # (a grid of pixel values — height × width × channels)
            img = mpimg.imread(str(pair["sample"]))

            # imshow() displays that array as an image on the axes
            self.ax_sample.imshow(img)

            # Show concise tag+index (e.g. "img_3" or "ndvi_0") using the parsed info
            si = pair.get("sample_info")
            if si is not None and si.get("tag") != "?":
                title_name = f"{si['tag']}_{si['index']}"
            else:
                title_name = pair['sample'].name

            self.ax_sample.set_title(
                f"samples/  {title_name}",
                color="#99ddff",        # light blue
                fontsize=9,
                fontfamily="monospace",
                pad=6
            )
        else:
            # Show a placeholder message when the file is missing
            self.ax_sample.text(
                0.5, 0.5, "FILE MISSING",
                ha="center", va="center",
                transform=self.ax_sample.transAxes,   # coordinates relative to axes (0–1)
                color="#ff4444", fontsize=16, fontweight="bold"
            )
            self.ax_sample.set_title(
                "samples/  — MISSING —",
                color="#ff6666", fontsize=9, fontfamily="monospace", pad=6
            )

        # -- LABEL PANEL (RIGHT) ----------------------------------------------
        self.ax_label.cla()

        if pair["label"] is not None:
            img = mpimg.imread(str(pair["label"]))
            self.ax_label.imshow(img)

            li = pair.get("label_info")
            if li is not None and li.get("tag") != "?":
                title_name = f"{li['tag']}_{li['index']}"
            else:
                title_name = pair['label'].name

            self.ax_label.set_title(
                f"labels/  {title_name}",
                color="#ffd966",        # warm yellow
                fontsize=9,
                fontfamily="monospace",
                pad=6
            )
        else:
            self.ax_label.text(
                0.5, 0.5, "FILE MISSING",
                ha="center", va="center",
                transform=self.ax_label.transAxes,
                color="#ff4444", fontsize=16, fontweight="bold"
            )
            self.ax_label.set_title(
                "labels/  — MISSING —",
                color="#ff6666", fontsize=9, fontfamily="monospace", pad=6
            )

        # -- MISMATCH BORDER --------------------------------------------------
        # Change the border (spine) color of both axes when there's a mismatch
        border_color = "#ff6622" if pair["mismatch"] else "#444444"
        border_width = 2.5       if pair["mismatch"] else 0.8

        # ax.spines is a dict of the four borders: "top", "bottom", "left", "right"
        for ax in (self.ax_sample, self.ax_label):
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)

        # fig.canvas.draw_idle() queues a redraw of the figure.
        # "idle" means it waits for the event loop to be free (efficient in notebooks).
        self.fig.canvas.draw_idle()

    # -------------------------------------------------------------------------
    #  _go  —  navigate to a specific index (with bounds clamping)
    # -------------------------------------------------------------------------

    def _go(self, index: int):
        """
        Navigates to pair at `index`, clamping to the valid range [0, n-1].

        Parameters
        ----------
        index : int
            The target index. Values outside [0, n-1] are clamped automatically.
        """

        # max(0, ...) ensures we never go below 0
        # min(self.n - 1, ...) ensures we never exceed the last index
        self.current = max(0, min(self.n - 1, index))
        self._render()

    # -------------------------------------------------------------------------
    #  _jump_mismatch  —  jump to the next or previous mismatch
    # -------------------------------------------------------------------------

    def _jump_mismatch(self, direction: int):
        """
        Jumps to the nearest mismatch in the given direction, wrapping around.

        Parameters
        ----------
        direction : int
            +1 to jump forward, -1 to jump backward.
        """

        # If there are no mismatches at all, do nothing
        if not self.mismatch_indices:
            return

        if direction > 0:
            # Find all mismatch indices that are *after* the current position
            candidates = [i for i in self.mismatch_indices if i > self.current]

            # If none found ahead, wrap around to the first mismatch
            next_idx = candidates[0] if candidates else self.mismatch_indices[0]
        else:
            # Find all mismatch indices that are *before* the current position
            candidates = [i for i in self.mismatch_indices if i < self.current]

            # If none found behind, wrap around to the last mismatch
            next_idx = candidates[-1] if candidates else self.mismatch_indices[-1]

        self._go(next_idx)


# =============================================================================
#  SECTION 5 — ENTRY POINT
#  This block only runs when you execute this file directly as a script
#  (e.g. `python image_viewer.py`), NOT when it's imported as a module.
#
#  The special variable __name__ equals "__main__" only during direct execution.
#  When you do `from image_viewer import ImageViewer` in a notebook,
#  __name__ equals "image_viewer" and this block is skipped.
# =============================================================================

if __name__ == "__main__":
    viewer = ImageViewer()
    viewer.show()
    input("Press Enter to close...")   # keeps the window open when run as a script
