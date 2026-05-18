"""
Generate a self-contained HTML viewer for a LoRA/FT dataset (data.jsonl).

Author: Adam Zvara (xzvara01)
Date: 04/2026
"""

import json
import os
import tempfile
from pathlib import Path

import mlflow


def _instr_bucket(n: int) -> str:
    if n < 80:
        return "short"
    if n < 250:
        return "medium"
    return "long"


def _lines_bucket(n: int) -> str:
    if n <= 5:
        return "short"
    if n <= 15:
        return "medium"
    return "long"


def _build_html(records: list[dict]) -> str:
    enriched = []
    for i, r in enumerate(records):
        instr = r.get("instruction", "")
        output = r.get("output", "")
        out_lines = output.count("\n") + 1 if output.strip() else 0
        enriched.append({
            "idx": i + 1,
            "instruction": instr,
            "output": output,
            "instr_len": len(instr),
            "instr_bucket": _instr_bucket(len(instr)),
            "out_lines": out_lines,
            "lines_bucket": _lines_bucket(out_lines),
        })

    return (
        """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dataset Viewer</title>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #f5f5f5; color: #222; padding: 1.5rem; }
  h1 { font-size: 1.3rem; margin-bottom: 1rem; }

  /* ── Toolbar ── */
  #toolbar {
    display: flex; gap: .75rem 1.25rem; flex-wrap: wrap;
    align-items: center; margin-bottom: .75rem;
    background: #fff; border: 1px solid #e0e0e0;
    border-radius: 10px; padding: .75rem 1rem;
  }
  #toolbar label { font-size: .8rem; font-weight: 700; color: #555; }
  #toolbar select, #toolbar input[type=text] {
    padding: .3rem .55rem; border-radius: 6px; border: 1px solid #ccc;
    background: #fafafa; font-size: .82rem;
  }
  #toolbar input[type=text] { width: 16rem; }
  .filter-sep { width: 1px; height: 1.4rem; background: #ddd; }
  #count { font-size: .82rem; color: #888; margin-left: auto; }

  /* ── Cards ── */
  #cards { display: flex; flex-direction: column; gap: 1rem; }
  .card {
    background: #fff; border-radius: 10px; border: 1px solid #e0e0e0;
    padding: 1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }

  /* ── Card header ── */
  .card-header {
    display: flex; gap: .45rem; align-items: center;
    margin-bottom: .9rem; flex-wrap: wrap;
  }
  .badge {
    font-size: .72rem; font-weight: 700; padding: .18rem .5rem;
    border-radius: 99px; white-space: nowrap;
  }
  .badge-id     { background: #f3f4f6; color: #555; }
  .badge-short  { background: #dcfce7; color: #166534; }
  .badge-medium { background: #fef9c3; color: #854d0e; }
  .badge-long   { background: #fee2e2; color: #b91c1c; }

  /* ── Sections ── */
  .section { margin-bottom: .8rem; }
  .section:last-child { margin-bottom: 0; }
  .section-label {
    font-size: .68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: #9ca3af; margin-bottom: .3rem;
  }
  .instruction-text {
    font-size: .85rem; white-space: pre-wrap; background: #fafafa;
    border: 1px solid #eee; border-radius: 6px; padding: .6rem .8rem;
    line-height: 1.55;
  }
  .output-block pre {
    border-radius: 7px; font-size: .8rem; overflow-x: auto; margin: 0;
  }
  .highlight-match { background: #fef08a; color: #000; border-radius: 2px; }
</style>
</head>
<body>
<h1>Dataset Viewer</h1>

<div id="toolbar">
  <label for="search-input">Search</label>
  <input type="text" id="search-input" placeholder="keyword in instruction…">

  <div class="filter-sep"></div>

  <label for="instr-filter">Instruction length</label>
  <select id="instr-filter">
    <option value="">All</option>
    <option value="short">Short (&lt;80 chars)</option>
    <option value="medium">Medium (80–249)</option>
    <option value="long">Long (250+)</option>
  </select>

  <div class="filter-sep"></div>

  <label for="lines-filter">Output lines</label>
  <select id="lines-filter">
    <option value="">All</option>
    <option value="short">Short (≤5)</option>
    <option value="medium">Medium (6–15)</option>
    <option value="long">Long (16+)</option>
  </select>

  <span id="count"></span>
</div>

<div id="cards"></div>

<script>
const RECORDS = """
        + json.dumps(enriched, ensure_ascii=False).replace('</', '<\/')
        + """;

// ── Escape HTML ────────────────────────────────────────────────────────────────
function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Highlight keyword matches in plain text ────────────────────────────────────
function highlightKeyword(text, kw) {
  if (!kw) return esc(text);
  const re = new RegExp(kw.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&'), 'gi');
  return esc(text).replace(re, m => `<mark class="highlight-match">${m}</mark>`);
}

// ── Build one card ─────────────────────────────────────────────────────────────
function makeCard(r) {
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.instrBucket = r.instr_bucket;
  card.dataset.linesBucket = r.lines_bucket;
  card.dataset.instruction = r.instruction.toLowerCase();

  // Header
  const hdr = document.createElement('div');
  hdr.className = 'card-header';

  function badge(text, cls) {
    const s = document.createElement('span');
    s.className = `badge ${cls}`;
    s.textContent = text;
    return s;
  }

  hdr.appendChild(badge(`#${r.idx}`, 'badge-id'));
  hdr.appendChild(badge(`instr: ${r.instr_bucket} (${r.instr_len} chars)`, `badge-${r.instr_bucket}`));
  hdr.appendChild(badge(`output: ${r.lines_bucket} (${r.out_lines} lines)`, `badge-${r.lines_bucket}`));
  card.appendChild(hdr);

  function section(label, child) {
    const sec = document.createElement('div');
    sec.className = 'section';
    const lbl = document.createElement('div');
    lbl.className = 'section-label';
    lbl.textContent = label;
    sec.appendChild(lbl);
    sec.appendChild(child);
    return sec;
  }

  // Instruction
  const instrEl = document.createElement('div');
  instrEl.className = 'instruction-text';
  card.appendChild(section('Instruction', instrEl));
  card._instrEl = instrEl;
  card._instrText = r.instruction;

  // Output (syntax-highlighted Python)
  const outWrap = document.createElement('div');
  outWrap.className = 'output-block';
  const pre = document.createElement('pre');
  const code = document.createElement('code');
  code.className = 'language-python';
  code.textContent = r.output;
  pre.appendChild(code);
  outWrap.appendChild(pre);
  card.appendChild(section('Output', outWrap));

  return card;
}

// ── Render all cards ───────────────────────────────────────────────────────────
const container = document.getElementById('cards');
const cards = RECORDS.map(makeCard);
cards.forEach(c => container.appendChild(c));
hljs.highlightAll();

// ── Filter & search ───────────────────────────────────────────────────────────
function applyFilters() {
  const kw    = document.getElementById('search-input').value.trim().toLowerCase();
  const instr = document.getElementById('instr-filter').value;
  const lines = document.getElementById('lines-filter').value;

  let visible = 0;
  cards.forEach(c => {
    const matchKw    = !kw    || c.dataset.instruction.includes(kw);
    const matchInstr = !instr || c.dataset.instrBucket === instr;
    const matchLines = !lines || c.dataset.linesBucket === lines;
    const show = matchKw && matchInstr && matchLines;
    c.style.display = show ? '' : 'none';

    // Update keyword highlighting in instruction text
    if (show) {
      c._instrEl.innerHTML = highlightKeyword(c._instrText, kw);
      visible++;
    }
  });
  document.getElementById('count').textContent = `${visible} / ${cards.length} shown`;
}

document.getElementById('search-input').addEventListener('input', applyFilters);
document.getElementById('instr-filter').addEventListener('change', applyFilters);
document.getElementById('lines-filter').addEventListener('change', applyFilters);
applyFilters();
</script>
</body>
</html>
"""
    )


def log_dataset_html(run_dir: Path) -> None:
    """Build a self-contained HTML viewer for data.jsonl and log it as an artifact."""
    # Try per-run data.jsonl first, then experiment-level
    path = run_dir / "data.jsonl"
    if not path.exists():
        path = run_dir.parent / "data.jsonl"
    if not path.exists():
        return

    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if not records:
        return

    html = _build_html(records)

    tmpdir = tempfile.mkdtemp()
    tmp_path = Path(tmpdir) / "dataset.html"
    tmp_path.write_text(html, encoding="utf-8")

    mlflow.log_artifact(str(tmp_path), artifact_path="html")
    tmp_path.unlink()
    os.rmdir(tmpdir)
