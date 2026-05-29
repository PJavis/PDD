# Getting Started — A Step-by-Step Guide for Total Beginners

This guide walks you through running the **Zinc Dendrite Simulator** from a completely empty computer. No prior coding experience needed. Just follow each step in order, and copy-paste the commands exactly.

> **What is this project?**
> A battery's metal part can grow tiny tree-like spikes called *dendrites*. If they grow too much, they can break the battery. This program is a tiny science lab on your computer: it draws how those metal trees grow, in pictures. You can change the settings (like temperature of the experiment) with sliders and watch the picture change.

---

## Table of Contents

1. [What you need](#1-what-you-need)
2. [Step 1 — Install a code editor (VS Code)](#step-1--install-a-code-editor-vs-code)
3. [Step 2 — Install `uv` (the tool that runs everything)](#step-2--install-uv-the-tool-that-runs-everything)
4. [Step 3 — Get the project onto your computer](#step-3--get-the-project-onto-your-computer)
5. [Step 4 — Run the easy web version (recommended first)](#step-4--run-the-easy-web-version-recommended-first)
6. [Step 5 — Run the notebook version (to see the science)](#step-5--run-the-notebook-version-to-see-the-science)
7. [What each file does](#what-each-file-does)
8. [What each slider does](#what-each-slider-does)
9. [If something goes wrong](#if-something-goes-wrong)

---

## 1. What you need

- A computer (Windows, Mac, or Linux).
- Internet connection (only for installing — after that it works offline).
- About 20 minutes for the first setup.

That's it. You do **not** need to know Python.

---

## Step 1 — Install a code editor (VS Code)

A code editor is like a special notepad for programs. We use the free one called **Visual Studio Code** (VS Code).

1. Open your web browser.
2. Go to: **https://code.visualstudio.com**
3. Click the big blue **Download** button. It picks the right version for your computer automatically.
4. Open the file you downloaded and click **Next / Install** until it finishes.
   - **Windows:** during install, tick the box that says *"Add to PATH"* if you see it.
5. Open VS Code once to make sure it starts.

> **Why?** VS Code gives you a built-in "terminal" (a black box where you type commands) and lets you open the project files. You could use any terminal, but VS Code keeps everything in one window.

### Open the terminal inside VS Code

- In VS Code, click the top menu: **Terminal → New Terminal**.
- A panel opens at the bottom. This is where you type the commands in the next steps.

---

## Step 2 — Install `uv` (the tool that runs everything)

`uv` is a small program that automatically downloads Python and all the science libraries this project needs. You install it **once**.

Copy the command for your system into the VS Code terminal, press **Enter**, and wait for it to finish.

### Windows
Paste this into the terminal (use the **PowerShell** terminal, which is the default):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Mac or Linux
Paste this:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After it finishes, **close the terminal and open a new one** (Terminal → New Terminal). This makes the computer notice the new tool.

### Check it worked
Type this and press Enter:
```bash
uv --version
```
You should see something like `uv 0.5.x`. If you see a version number, you're good. ✅

> **Why `uv`?** Normally you'd have to install Python, then install many libraries one by one, and they often fight each other. `uv` does all of that for you automatically and never messes up the rest of your computer.

---

## Step 3 — Get the project onto your computer

You need the project files. Pick **ONE** of these two ways.

### Way A — Download a ZIP (easiest, no extra tools)
1. Go to the project page: **https://github.com/PJavis/PDD**
2. Click the green **`< > Code`** button → **Download ZIP**.
3. Find the downloaded `PDD-main.zip`, right-click → **Extract All** (Windows) or double-click (Mac).
4. You now have a folder called `PDD-main` (or `PDD`). Remember where it is (e.g. your Desktop).

### Way B — Use `git` (if you already have git)
In the terminal:
```bash
git clone https://github.com/PJavis/PDD.git
```

### Open the folder in VS Code
1. In VS Code: **File → Open Folder…**
2. Choose the project folder (`PDD` or `PDD-main`).
3. Open a fresh terminal (**Terminal → New Terminal**). It now starts *inside* the project folder — this is important. The bottom of the terminal should show the folder name.

> **How do I know I'm in the right place?** Type `ls` (Mac/Linux) or `dir` (Windows) and press Enter. You should see file names like `app.py`, `fd_core.py`, `README.md`. If you see those, you're in the right folder. ✅

---

## Step 4 — Run the easy web version (recommended first)

This opens a webpage with sliders, right on your own computer.

1. In the project terminal, paste this and press Enter:
   ```bash
   uv run --with numpy --with matplotlib --with gradio python3 app.py
   ```
2. The **first time**, it downloads the science libraries. This takes 1–3 minutes. You'll see lots of text scroll by — that's normal.
3. When it's ready, you'll see a line like:
   ```
   Running on local URL:  http://127.0.0.1:7860
   ```
4. Hold **Ctrl** (or **Cmd** on Mac) and click that link, or copy it into your browser.
5. The simulator opens. Move the sliders, click **Run simulation**, and watch the pictures appear. 🎉

> **First run takes ~30–60 seconds to compute** after you click Run. Be patient — it's doing real physics math.

### To stop it
Click in the terminal and press **Ctrl + C**. This shuts the webpage down.

---

## Step 5 — Run the notebook version (to see the science)

A "notebook" mixes explanations, code, and result pictures in one scrollable page. Good for understanding *what* and *why*.

1. In the project terminal, paste:
   ```bash
   uv run --with numpy --with matplotlib --with jupyter python3 -m jupyter notebook fd_zinc_demo.ipynb
   ```
2. Your browser opens automatically showing the notebook.
3. To run everything: click the top menu **Run → Run All Cells**.
4. Scroll down. You'll see:
   - The metal tree growing (yellow blob with spikes).
   - The "ion" map showing where the metal food gets used up.
   - The electric field map.
   - Graphs of how tall the tree gets over time.

> **Tip:** each gray box is a "cell." Click a cell and press **Shift + Enter** to run just that one.

### To stop it
Go back to the terminal and press **Ctrl + C**, then type `y` and Enter if asked.

---

## What each file does

| File | What it is | Do I touch it? |
|------|-----------|----------------|
| `app.py` | The web version with sliders | No — just run it (Step 4) |
| `fd_zinc_demo.ipynb` | The notebook with explanations | No — just run it (Step 5) |
| `fd_core.py` | The "engine" that does the physics math | Only if you want to learn the code |
| `fd_core_numba.py` | A faster copy of the engine | Optional, for speed |
| `README.md` | The technical summary for grown-up programmers | Read later |
| `Mô phỏng pin kẽm dung môi nước.pdf` | The science paper this is based on | Reference reading |

---

## What each slider does

In the web version (Step 4), you'll see these sliders. Here's what they mean in plain words:

| Slider | Plain-language meaning | Try this |
|--------|------------------------|----------|
| **k_dep** | How *fast* the metal sticks. Higher = faster, bushier trees. | Slide it high (20+) to see big bushy growth |
| **Ds** | How easily the metal "food" (ions) moves around. | Higher = trees grow taller |
| **E_theta** | How hard you push the electricity. More negative = stronger push. | Try -0.4 for faster growth |
| **u_inlet** | Wind speed of liquid flowing past. 0 = still water. | Set to 0.06 and watch the tree lean sideways! |
| **steps** | How long to run the experiment. More = taller tree but slower. | Start at 4000 |
| **Polycrystalline** | Tick this to grow *many* trees at once instead of one. | Tick it to see trees compete |
| **n_seeds** | How many trees (only if Polycrystalline is ticked). | Try 6–8 |

After moving sliders, always click **Run simulation** again to see the new result.

---

## If something goes wrong

**"uv: command not found" or "uv is not recognized"**
→ You skipped closing/reopening the terminal in Step 2. Close all terminals, open a new one, try `uv --version` again. If still broken, redo Step 2.

**"No such file or directory: app.py"**
→ Your terminal isn't inside the project folder. Redo "Open the folder in VS Code" in Step 3. Type `ls` (or `dir`) and confirm you see `app.py`.

**The first run is very slow / seems frozen**
→ Normal. The first run downloads libraries (1–3 min) and the first calculation takes ~30–60 sec. Wait. Text scrolling = it's working.

**The webpage link doesn't open**
→ Copy the address (e.g. `http://127.0.0.1:7860`) and paste it into your browser's address bar by hand.

**I want to start completely over**
→ Delete the project folder, redo Step 3. Your `uv` and VS Code installs stay — you only redo the download.

**Still stuck?**
→ Copy the red error text from the terminal and ask for help (a teacher, or paste it into a search engine). The exact words matter.

---

## What's next?

Once you're comfortable:
- Open `fd_core.py` in VS Code and read the comments — they explain each physics equation.
- Change a number in the `run(...)` defaults and re-run to see what happens.
- Read `README.md` for the technical details and the science paper for the real research.

Have fun growing metal trees! 🌱⚡
