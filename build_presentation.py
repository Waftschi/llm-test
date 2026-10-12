#!/usr/bin/env python3
"""Build one standalone HTML file from a deck folder (project/deck.json + project/slides/*.html).

    python3 build_presentation.py praesentation      -> praesentation/presentation.html
    python3 build_presentation.py praesentation_en   -> praesentation_en/presentation.html

Open the result in a browser. Keys: arrows / space = next and previous, F = full screen,
N = speaker notes, Home / End = first and last slide.
Only the fonts come from the internet (Google Fonts); without a connection a system font is used.
"""
import html, json, re, sys
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "praesentation")
deck = json.loads((folder / "project" / "deck.json").read_text(encoding="utf-8"))

slides, notes = [], []
for sid in deck["order"]:
    text = (folder / "project" / "slides" / f"{sid}.html").read_text(encoding="utf-8")
    m = re.search(r"<aside>(.*?)</aside>", text, flags=re.S)
    notes.append(html.unescape(m.group(1)).strip() if m else "")
    slides.append(re.sub(r"<aside>.*?</aside>", "", text, flags=re.S).strip())

fonts = "".join(
    f'<link rel="stylesheet" href="{f["href"]}">' for f in deck.get("faces", {}).values() if "href" in f
)
title = html.escape(deck["title"])

page = """<!DOCTYPE html>
<html lang="__LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__FONTS__
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: #0e0c0a; overflow: hidden; }
  body { display: flex; flex-direction: column; font-family: 'DM Sans', Arial, sans-serif; }
  #stage { position: relative; flex: 1; min-height: 0; overflow: hidden; }
  #canvas { position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; transform-origin: 0 0; overflow: hidden; }
  #canvas section { position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; overflow: hidden; }
  #canvas section:not(.active) { display: none !important; }
  #canvas p, #canvas h1, #canvas h2, #canvas h3, #canvas ul, #canvas ol { margin: 0; }
  #canvas ul, #canvas ol { padding-left: 48px; }
  #canvas table { width: 100%; border-collapse: collapse; }
  #canvas th, #canvas td { padding: 0.35em 0.6em; text-align: left; border-bottom: 1px solid rgba(128,128,128,0.35); }
  #canvas hr { width: 100%; margin: 0; border: 0; }
  #canvas x-shape { display: block; flex-shrink: 0; }
  #canvas x-shape[kind="arrow-right"] { clip-path: polygon(0 30%, 60% 30%, 60% 0, 100% 50%, 60% 100%, 60% 70%, 0 70%); }
  #canvas x-shape[kind="arrow-left"] { clip-path: polygon(100% 30%, 40% 30%, 40% 0, 0 50%, 40% 100%, 40% 70%, 100% 70%); }
  #bar { display: flex; justify-content: space-between; align-items: center; padding: 6px 14px;
         color: #8a8174; font-size: 14px; background: #0e0c0a; }
  #bar button { background: none; border: 1px solid #4a443b; color: #bdb5a6; border-radius: 6px; padding: 2px 10px; cursor: pointer; font-size: 14px; }
  #notes { display: none; max-height: 26vh; overflow-y: auto; padding: 14px 24px; background: #1e1b18; color: #f6f1e7;
           font-size: 18px; line-height: 1.5; border-top: 1px solid #4a443b; }
  body.notes #notes { display: block; }
</style>
</head>
<body>
<div id="stage"><div id="canvas">
__SLIDES__
</div></div>
<div id="notes"></div>
<div id="bar">
  <span id="count"></span>
  <span>
    <button id="bn">__NOTES__ (N)</button>
    <button id="bf">__FULL__ (F)</button>
    <button id="bp">&#9664;</button>
    <button id="bx">&#9654;</button>
  </span>
</div>
<script>
const NOTES = __NOTES_JSON__;
const secs = Array.from(document.querySelectorAll('#canvas section'));
const stage = document.getElementById('stage'), canvas = document.getElementById('canvas');
let cur = 0;
function fit() {
  const s = Math.min(stage.clientWidth / 1920, stage.clientHeight / 1080);
  canvas.style.transform = 'scale(' + s + ')';
  canvas.style.left = ((stage.clientWidth - 1920 * s) / 2) + 'px';
  canvas.style.top = ((stage.clientHeight - 1080 * s) / 2) + 'px';
}
function show(i) {
  cur = Math.max(0, Math.min(secs.length - 1, i));
  secs.forEach((s, k) => s.classList.toggle('active', k === cur));
  document.getElementById('count').textContent = (cur + 1) + ' / ' + secs.length + '  ·  ' + document.title;
  document.getElementById('notes').textContent = NOTES[cur] || '';
  history.replaceState(null, '', '#' + (cur + 1));
  fit();
}
function toggleNotes() { document.body.classList.toggle('notes'); fit(); }
function toggleFull() {
  if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen();
}
document.addEventListener('keydown', e => {
  if (['ArrowRight', 'ArrowDown', 'PageDown', ' ', 'Enter'].includes(e.key)) { e.preventDefault(); show(cur + 1); }
  else if (['ArrowLeft', 'ArrowUp', 'PageUp', 'Backspace'].includes(e.key)) { e.preventDefault(); show(cur - 1); }
  else if (e.key === 'Home') show(0);
  else if (e.key === 'End') show(secs.length - 1);
  else if (e.key === 'n' || e.key === 'N') toggleNotes();
  else if (e.key === 'f' || e.key === 'F') toggleFull();
});
stage.addEventListener('click', e => show(e.clientX > stage.clientWidth / 2 ? cur + 1 : cur - 1));
document.getElementById('bn').onclick = toggleNotes;
document.getElementById('bf').onclick = toggleFull;
document.getElementById('bp').onclick = () => show(cur - 1);
document.getElementById('bx').onclick = () => show(cur + 1);
window.addEventListener('resize', fit);
show((parseInt(location.hash.slice(1), 10) || 1) - 1);
</script>
</body>
</html>
"""

german = "praesentation" in folder.name and not folder.name.endswith("_en")
out = (page.replace("__LANG__", "de" if german else "en")
           .replace("__TITLE__", title)
           .replace("__FONTS__", fonts)
           .replace("__SLIDES__", "\n".join(slides))
           .replace("__NOTES__", "Notizen" if german else "Notes")
           .replace("__FULL__", "Vollbild" if german else "Full screen")
           .replace("__NOTES_JSON__", json.dumps(notes, ensure_ascii=False).replace("</", "<\\/")))
target = folder / "presentation.html"
target.write_text(out, encoding="utf-8")
print(f"{len(slides)} slides -> {target}")
