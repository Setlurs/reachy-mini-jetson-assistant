"""Render the Big-Brain-Dev / Small-Brain-Ops architecture to a PDF.

Narrative: the entire system was designed, implemented, tested and
documented efficiently by Claude Code (the "big brain", at dev time);
it runs as a small, secure, on-prem local stack (the "small brain", at
ops time) that works offline.

Run: venv/bin/python docs/make_architecture_pdf.py  ->  docs/architecture.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_TXT = "#e8eef5"
C_DIM = "#9fb3c8"
C_DEV = "#3a2350"          # big brain / dev = purple
C_DEV_BD = "#b083e0"
C_OPS = "#16324f"          # small brain / ops = blue
C_OPS_BD = "#4f9bd9"
C_GREEN = "#173a2b"
C_GREEN_BD = "#3fae74"
C_BOX = "#0f1722"
C_BOX_BD = "#3a4a5a"

fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor("#0a0e14")
ax.set_facecolor("#0a0e14")
ax.set_xlim(0, 160)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, lines=None, fc=C_BOX, ec=C_BOX_BD,
        tcolor=C_TXT, fs=8.5, tfs=10):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.4",
        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="top",
            fontsize=tfs, fontweight="bold", color=tcolor, zorder=3)
    if lines:
        ax.text(x + w / 2, y + h - 7.0, "\n".join(lines), ha="center",
                va="top", fontsize=fs, color=C_DIM, zorder=3,
                linespacing=1.5)


def region(x, y, w, h, label, sub, fc, ec):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
        linewidth=2, edgecolor=ec, facecolor=fc, alpha=0.35, zorder=1))
    ax.text(x + 3, y + h - 2.6, label, ha="left", va="top",
            fontsize=13, fontweight="bold", color=ec, zorder=3)
    ax.text(x + 3 + 34, y + h - 3.0, sub, ha="left", va="top",
            fontsize=9.5, color=ec, zorder=3)


def chip(x, y, w, h, text, ec=C_OPS_BD):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.3",
        linewidth=1.2, edgecolor=ec, facecolor="#13202d", zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=8, color=C_TXT, zorder=3)


def arrow(x1, y1, x2, y2, color=C_OPS_BD, style="-|>", lw=2.0, rad=0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=18,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=4))


# ── Title ─────────────────────────────────────────────────────────
ax.text(80, 97, "Reachy Mini Assistant  —  Built by Claude Code, Runs On-Prem",
        ha="center", fontsize=18, fontweight="bold", color=C_TXT)
ax.text(80, 92.6,
        "Big-Brain DEV (Claude Code) builds it efficiently   ·   "
        "Small-Brain OPS runs it securely, locally, offline",
        ha="center", fontsize=11.5, color=C_DIM)

# ── BIG BRAIN · DEV (Claude Code) ─────────────────────────────────
region(4, 58, 152, 31, "BIG BRAIN · DEV", "", C_DEV, C_DEV_BD)

box(8, 62, 40, 23, "Claude Code",
    ["agentic software developer",
     "- plans & writes the code",
     "- refactors / fixes bugs",
     "- runs & verifies tests",
     "- commits & pushes to git",
     "- writes docs (this PDF)"],
    fc="#2c1e3e", ec=C_DEV_BD, tfs=11)

ax.text(52, 84.5, "Delivered  —  whole system, iteratively & efficiently:",
        ha="left", fontsize=9.5, color=C_DEV_BD, fontweight="bold")
_chips = [
    "--local-media backend\n(mic / cam / speaker)",
    "deterministic intercepts\n(mic / cam / video / status)",
    "plug-in tools\n(camera, mic, video, time)",
    "web UI :8090\n(text query, toggles, video)",
    "wake-word unmute\n(on-device tflite)",
    "gapless TTS pipeline\n(crackle fix)",
    "conversation history\n+ text-tool-call recovery",
    "config + this\narchitecture PDF",
]
cx, cy, cw, ch = 52, 62, 24.5, 9.5
for i, t in enumerate(_chips):
    r, c = divmod(i, 4)
    chip(cx + c * (cw + 1.5), cy + (1 - r) * (ch + 1.5), cw, ch, t,
         ec=C_DEV_BD)

ax.text(80, 60.4,
        "Weeks of engineering compressed; every step test-verified and "
        "version-controlled.",
        ha="center", fontsize=8.5, color=C_DIM, style="italic")

# ── builds / ships arrow ──────────────────────────────────────────
arrow(80, 57.5, 80, 51.5, color=C_DEV_BD, lw=2.6)
ax.text(82, 54.3, "builds · ships · maintains", ha="left", fontsize=9,
        color=C_DEV_BD, va="center")

# ── SMALL BRAIN · OPS (secure on-prem runtime) ────────────────────
region(4, 6, 152, 45, "SMALL BRAIN · OPS",
       "secure on-prem runtime  —  local, offline-capable, no cloud",
       C_OPS, C_OPS_BD)

# pipeline row
py, ph = 30, 14
box(8, py, 20, ph, "USER",
    ["voice / typed", "web UI :8090"], fc="#22303f", ec="#6da9d6")
box(30, py, 22, ph, "MEDIA I/O",
    ["pluggable:", "Reachy | local", "(sounddevice/CV)"], ec=C_OPS_BD)
box(54, py, 30, ph, "PERCEPTION",
    ["VAD (Silero)", "wake-word tflite", "STT faster-whisper"], ec=C_OPS_BD)
box(86, py, 30, ph, "DETERMINISTIC\nINTERCEPTS",
    ["mic / camera / video", "status  —  instant", "no LLM · no network"],
    fc=C_GREEN, ec=C_GREEN_BD)
box(118, py, 34, ph, "REASONING (only if needed)",
    ["local LLM/VLM  Ollama", "+ tool registry (plug-in)", "history · vision"],
    fc="#1d2b3a", ec=C_OPS_BD)

box(8, 9, 60, 16, "OUTPUT",
    ["TTS Kokoro/XTTS (local, pluggable) -> speaker (gapless)",
     "Web UI: tokens · transcript · map · video panel",
     "emotion -> motor (robot)"],
    fc="#15212e", ec=C_OPS_BD)

box(72, 9, 50, 16, "SECURE / ON-PREM",
    ["runs on the device / LAN  —  no cloud, no telemetry",
     "data never leaves the box  ·  airplane-mode capable",
     "only optional tools (search/APIs) touch the network"],
    fc=C_GREEN, ec=C_GREEN_BD)

box(126, 9, 26, 16, "EXTENSIBLE",
    ["Claude Code adds a", "tool / backend / clip",
     "in minutes; config-", "driven (settings.yaml)"],
    fc="#1a2435", ec="#7aa7c7", fs=8)

# pipeline flow arrows
for x in (28, 52, 84, 116):
    arrow(x, py + ph / 2, x + 2, py + ph / 2, color=C_OPS_BD, lw=1.6)
arrow(100, py, 100, 25.5, color=C_GREEN_BD, lw=1.6)          # reflex down
ax.text(101.5, 27.5, "reflex", ha="left", fontsize=7.5, color=C_GREEN_BD)
arrow(135, py, 120, 25.5, color=C_OPS_BD, lw=1.6, rad=0.15)  # reasoning down

ax.text(80, 2.0,
        "Same conceptual split, two timescales:  the big brain (Claude "
        "Code) does the thinking-heavy build;  the small brain runs lean, "
        "private, and offline in production.",
        ha="center", fontsize=8.5, color=C_DIM, style="italic")

plt.tight_layout()
out = "docs/architecture.pdf"
fig.savefig(out, format="pdf", facecolor=fig.get_facecolor(),
            bbox_inches="tight")
print("wrote", out)
