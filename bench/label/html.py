"""Generate the self-contained ``label.html`` (inline JS/CSS, base64 views, no server)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from bench.label.io import IdentityLabels, load_labels, seed_from_result
from countonce.explain import load_result_dir

MAX_PX = 900


def _find_images(out_dir: Path) -> list[Path]:
    """Source views of a run: ``sanitized/**`` images, in sorted order (index = frame id)."""
    root = out_dir / "sanitized"
    if not root.is_dir():
        return []
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def _data_uri(path: Path) -> tuple[str, int, int]:
    from PIL import Image

    with Image.open(path) as src:
        im = src.convert("RGB")
        w, h = im.size
        im.thumbnail((MAX_PX, MAX_PX))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), w, h


def build_views(out_dir: Path) -> tuple[list[dict[str, Any]], Any]:
    """Per-view image + detection points (original-image pixels) from a finished run.

    ``result.json`` stores each detection's centre in recon-grid pixels, not its box, so the
    labeler marks detections as circles at that centre. The grid size is inferred from the camera
    principal point (2*cx, 2*cy) and rescaled to the image; that is exact to about one pixel.
    """
    result = load_result_dir(out_dir)
    images = _find_images(out_dir)
    cams = {c.view: c for c in result.cameras}
    frames = sorted({d.frame_id for d in result.detections} | set(range(len(images))))
    views: list[dict[str, Any]] = []
    for fid in frames:
        uri, w, h = (None, 0, 0)
        if fid < len(images):
            uri, w, h = _data_uri(images[fid])
        dets = [d for d in result.detections if d.frame_id == fid]
        cam = next((cams[d.view] for d in dets if d.view in cams), cams.get(fid))
        if cam is not None:
            gw, gh = 2 * cam.intrinsics[0][2], 2 * cam.intrinsics[1][2]
        else:
            gw = gh = 0.0
        if not w and dets:  # no image on disk: blank canvas the size of the recon grid
            w, h = int(gw) or 640, int(gh) or 480
        sx, sy = (w / gw, h / gh) if gw > 0 and gh > 0 else (1.0, 1.0)
        pts = [
            {"id": d.id, "x": round(d.uv[0] * sx, 1), "y": round(d.uv[1] * sy, 1)}
            for d in dets
            if d.uv is not None
        ]
        views.append({"frame": fid, "image": uri, "w": w, "h": h, "dets": pts})
    return views, result


_CSS = """
body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh;color:#222}
#main{flex:1;overflow:auto;padding:.6rem}#side{width:19rem;border-left:1px solid #ccc;
padding:.6rem;overflow:auto;font-size:.9rem}
.stage{position:relative;display:inline-block;border:1px solid #bbb}
.stage img,.stage .blank{display:block;max-width:100%;background:#eee}
svg{position:absolute;left:0;top:0;width:100%;height:100%}
circle{stroke-width:3;cursor:pointer;fill-opacity:.25}
circle.sel{stroke-dasharray:5 3}text{font-size:13px;pointer-events:none;fill:#000;
paint-order:stroke;stroke:#fff;stroke-width:3px}
button{margin:.15rem}.bar{height:.6rem;background:#ddd}.bar i{display:block;height:100%;
background:#2a7}kbd{background:#eee;border:1px solid #bbb;border-radius:3px;padding:0 .25rem}
#msg{color:#b00;min-height:1.2rem}li.act{font-weight:700}
"""

_JS = r"""
const D=JSON.parse(document.getElementById('data').textContent);
const PAL=['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#008080','#9a6324','#800000',
'#808000','#000075','#f032e6','#469990'];
let labels=Object.assign({},D.seed.labels),rej=new Set(D.seed.rejected),cur=0,active=null,
hist=[],counter=0,last=null;
const frameOf=id=>+id.split(':')[0];
for(const v of Object.values(labels)){const m=/^i?(\d+)$/.exec(v);
if(m)counter=Math.max(counter,+m[1]+1)}
const all=D.views.flatMap(v=>v.dets.map(d=>d.id));
const $=id=>document.getElementById(id);
const col=i=>PAL[Math.abs([...String(i)].reduce((a,c)=>a*31+c.charCodeAt(0)|0,7))%PAL.length];
function snap(){hist.push([JSON.stringify(labels),JSON.stringify([...rej]),active])}
function undo(){if(!hist.length)return;const[a,b,c]=hist.pop();labels=JSON.parse(a);
rej=new Set(JSON.parse(b));active=c;render()}
function msg(t){$('msg').textContent=t||''}
function newInst(){active='n'+(counter++);msg('');render()}
function members(i){return Object.keys(labels).filter(k=>labels[k]===i)}
function click(id,ev){
 last=id;
 if(ev.shiftKey){const t=labels[id];if(t===undefined)return msg('not labeled');
  if(active&&t!==active){snap();// merge t into active, checking cannot-link
   const fr=new Set(members(active).map(frameOf));
   if(members(t).some(m=>fr.has(frameOf(m))))return msg('merge refused: same-view detections');
   for(const m of members(t))labels[m]=active;}else active=t;render();return}
 if(rej.has(id)){snap();rej.delete(id);render();return}
 if(active===null){snap();active='n'+(counter++)}
 if(labels[id]===active){snap();delete labels[id];render();return}
 if(members(active).some(m=>frameOf(m)===frameOf(id)&&m!==id))
  return msg('cannot-link: instance '+active+' already has a detection in this view');
 snap();labels[id]=active;msg('');render()}
function reject(){if(!last)return;snap();delete labels[last];rej.add(last);render()}
function go(d){cur=Math.max(0,Math.min(D.views.length-1,cur+d));render()}
function render(){
 const v=D.views[cur];let h='<div class="stage">';
 h+=v.image?`<img src="${v.image}">`:
  `<div class="blank" style="width:${v.w}px;height:${v.h}px"></div>`;
 h+=`<svg viewBox="0 0 ${v.w} ${v.h}">`;
 const r=Math.max(8,Math.min(v.w,v.h)/40);
 for(const d of v.dets){const l=labels[d.id],x=rej.has(d.id);
  const c=x?'#777':l!==undefined?col(l):'#000';
  h+=`<circle data-id="${d.id}" cx="${d.x}" cy="${d.y}" r="${r}" stroke="${c}" fill="${c}"
   class="${l!==undefined&&l===active?'sel':''}"/>
   <text x="${d.x+r+2}" y="${d.y}">${x?'✕':l!==undefined?l:'?'}</text>`}
 h+='</svg></div>';
 $('view').innerHTML=h;
 document.querySelectorAll('circle').forEach(c=>c.onclick=e=>click(c.dataset.id,e));
 $('vi').textContent=`view ${cur+1}/${D.views.length} (frame ${v.frame})`;
 const done=all.filter(i=>labels[i]!==undefined||rej.has(i)).length;
 $('prog').textContent=`${done}/${all.length} detections handled`;
 $('bar').style.width=(all.length?100*done/all.length:100)+'%';
 const ids=[...new Set(Object.values(labels))].sort();
 $('inst').innerHTML=ids.map(i=>`<li class="${i===active?'act':''}" style="color:${col(i)}">
  ${i}: ${members(i).length} dets, ${new Set(members(i).map(frameOf)).size} views</li>`).join('');
 $('active').textContent=active===null?'(none)':active}
function exp(){
 const out={schema_version:'1',scene:D.scene,labels,rejected:[...rej].sort(),
  labeler:$('who').value,timestamp:new Date().toISOString()};
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
 a.download='identity_labels.json';a.click()}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
 if(e.key==='ArrowRight'||e.key==='d')go(1);else if(e.key==='ArrowLeft'||e.key==='a')go(-1);
 else if(e.key==='n')newInst();else if(e.key==='x')reject();
 else if(e.key==='u'||(e.key==='z'&&(e.ctrlKey||e.metaKey)))undo();
 else if(e.key==='Escape'){active=null;render()}
 else if(e.key==='s'&&(e.ctrlKey||e.metaKey)){e.preventDefault();exp()}});
$('prev').onclick=()=>go(-1);$('next').onclick=()=>go(1);$('undo').onclick=undo;
$('new').onclick=newInst;$('rej').onclick=reject;$('exp').onclick=exp;render();
"""


def render_html(scene: str, views: list[dict[str, Any]], seed: IdentityLabels) -> str:
    data = json.dumps({"scene": scene, "views": views, "seed": seed.to_dict()})
    data = data.replace("</", "<\\/")  # keep the JSON block from closing its <script>
    esc = scene.replace("&", "&amp;").replace("<", "&lt;")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CountOnce labeler: {esc}</title>
<style>{_CSS}</style></head><body>
<div id="main"><div><button id="prev">&larr; prev</button><button id="next">next &rarr;</button>
<span id="vi"></span></div><div id="msg"></div><div id="view"></div></div>
<div id="side"><h3>Identity labels: {esc}</h3>
<div class="bar"><i id="bar"></i></div><div id="prog"></div>
<p>Active instance: <b id="active"></b></p>
<button id="new">new instance (n)</button><button id="rej">not target (x)</button>
<button id="undo">undo (u)</button><br>
<label>labeler <input id="who" size="12"></label>
<button id="exp">export JSON (Ctrl+S)</button>
<h4>Keys</h4><ul><li><kbd>click</kbd> assign the active instance (new if none); click again to
unassign</li><li><kbd>shift+click</kbd> pick up an instance; with another active, merge into it
</li><li><kbd>n</kbd> new <kbd>x</kbd> not target (last clicked) <kbd>u</kbd> undo
<kbd>Esc</kbd> deselect</li><li><kbd>&larr; &rarr;</kbd> or <kbd>a d</kbd> change view</li></ul>
<h4>Instances</h4><ul id="inst"></ul></div>
<script id="data" type="application/json">{data}</script>
<script>{_JS}</script></body></html>
"""


def write_label_html(
    out_dir: Path | str,
    dest: Path | str | None = None,
    prelabel: bool = False,
    labels_path: Path | str | None = None,
) -> Path:
    """Write ``label.html`` for a finished run.

    Seeds: ``prelabel`` uses the run's own instances (so the human only fixes mistakes);
    ``labels_path`` resumes from a previously exported ``identity_labels.json`` instead.
    """
    out = Path(out_dir)
    views, result = build_views(out)
    scene = out.resolve().name
    if labels_path is not None:
        known = {d["id"] for v in views for d in v["dets"]}
        seed = load_labels(labels_path, known)
    elif prelabel:
        seed = seed_from_result(result, scene)
    else:
        seed = IdentityLabels(scene)
    path = Path(dest) if dest else out / "label.html"
    path.write_text(render_html(scene, views, seed), encoding="utf-8")
    return path
