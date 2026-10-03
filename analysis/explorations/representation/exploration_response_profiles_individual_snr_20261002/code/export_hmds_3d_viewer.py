"""Export an offline camera-only viewer for the saved 106-sample 3D HMDS fits."""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import plotly
from plotly.offline import get_plotlyjs


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D HMDS · camera viewer</title>
<style>
* { box-sizing:border-box; }
body { margin:0; background:#f5f7fa; color:#18324c; font:15px/1.5 system-ui,sans-serif; }
main { max-width:1540px; margin:auto; padding:26px 24px 32px; }
h1 { font-size:25px; margin:0 0 5px; letter-spacing:-.4px; }
.note { color:#607185; margin:0 0 18px; }
.toolbar { display:flex; flex-wrap:wrap; gap:9px; align-items:center; margin:16px 0; }
button { border:1px solid #bfcbd7; background:white; border-radius:7px; color:#18324c;
         padding:8px 13px; font:inherit; cursor:pointer; }
button:hover { background:#edf3f8; }
button:disabled { opacity:.5; cursor:wait; }
button.primary { background:#244d74; border-color:#244d74; color:white; }
button.primary:hover { background:#183d60; }
.plots { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }
.panel { background:white; border:1px solid #dce3eb; border-radius:11px; overflow:hidden; }
.panel-head { padding:14px 18px 0; display:flex; align-items:center; justify-content:space-between; gap:12px; }
h2 { font-size:18px; margin:0; font-weight:650; }
.count { color:#6b7c8e; font-size:13px; margin-left:8px; font-weight:400; }
.plot { width:100%; height:clamp(420px,47vw,630px); }
.panel button { font-size:13px; padding:6px 10px; }
.json-panel { margin-top:20px; padding:16px 18px 18px; background:white;
              border:1px solid #dce3eb; border-radius:11px; }
.json-panel .toolbar { margin:0 0 12px; }
.json-panel h2 { margin-right:auto; }
textarea { display:block; width:100%; height:240px; resize:vertical; border:1px solid #cbd5df;
           border-radius:6px; background:#fafbfd; color:#23394d; padding:12px;
           font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace; }
#viewer-status { margin-left:auto; font-size:13px; color:#607185; }
@media(max-width:850px) { main { padding:18px 12px; } .plots { grid-template-columns:1fr; }
 .plot { height:460px; } #viewer-status { width:100%; margin:0; } }
</style>
<script>__PLOTLY_JS__</script>
</head>
<body><main>
<h1>3D HMDS</h1>
<p class="note">Rotate each view independently. Only the camera changes; coordinates stay fixed.</p>
<div class="toolbar">
  <button id="reset-views" disabled>Reset both views</button>
  <button id="download-views" class="primary" disabled>Download views JSON</button>
  <span id="viewer-status" role="status" aria-live="polite">Loading saved coordinates…</span>
</div>
<section class="plots">
  <article class="panel"><div class="panel-head"><h2>Neural<span class="count">106 samples</span></h2>
    <button id="download-neural-png" disabled>Download PNG</button></div><div id="neural-plot" class="plot"></div></article>
  <article class="panel"><div class="panel-head"><h2>Chemical<span class="count">106 samples</span></h2>
    <button id="download-chemical-png" disabled>Download PNG</button></div><div id="chemical-plot" class="plot"></div></article>
</section>
<section class="json-panel">
 <div class="toolbar"><h2>Current views JSON</h2><button id="copy-views" disabled>Copy JSON</button></div>
 <textarea id="camera-json" readonly aria-label="Current camera parameters and provenance" spellcheck="false"></textarea>
</section>
</main>
<script id="hmds-viewer-data" type="application/json">__VIEWER_DATA__</script>
<script>
'use strict';
const DATA = JSON.parse(document.getElementById('hmds-viewer-data').textContent);
const DOMAINS = ['neural', 'chemical'];
const DEFAULT_CAMERA = {eye:{x:1.45,y:1.45,z:1.1}, up:{x:0,y:0,z:1},
                        center:{x:0,y:0,z:0}, projection:{type:'perspective'}};
const plots = Object.fromEntries(DOMAINS.map(d => [d, document.getElementById(`${d}-plot`)]));
const status = document.getElementById('viewer-status');
const textarea = document.getElementById('camera-json');
const clone = x => JSON.parse(JSON.stringify(x));
let ready = false;

function wireframe() {
  const x=[], y=[], z=[];
  const add = (xs,ys,zs) => {x.push(...xs,null);y.push(...ys,null);z.push(...zs,null);};
  const circle = Array.from({length:97}, (_,i) => 2*Math.PI*i/96);
  for (let latitude=-60; latitude<=60; latitude+=30) {
    const t=latitude*Math.PI/180, r=Math.cos(t);
    add(circle.map(a=>r*Math.cos(a)),circle.map(a=>r*Math.sin(a)),circle.map(()=>Math.sin(t)));
  }
  for (let longitude=0; longitude<180; longitude+=30) {
    const t=longitude*Math.PI/180;
    add(circle.map(a=>Math.cos(a)*Math.cos(t)),circle.map(a=>Math.cos(a)*Math.sin(t)),circle.map(a=>Math.sin(a)));
  }
  return {type:'scatter3d',mode:'lines',x,y,z,line:{color:'#acbac8',width:1},
          opacity:.24,hoverinfo:'skip',showlegend:false};
}
function layout() {
  const axis={range:[-1.04,1.04],autorange:false,visible:false,showgrid:false,zeroline:false};
  return {autosize:true,margin:{l:0,r:0,t:0,b:0},paper_bgcolor:'#ffffff',showlegend:false,
          scene:{xaxis:clone(axis),yaxis:clone(axis),zaxis:clone(axis),aspectmode:'cube',
                 aspectratio:{x:1,y:1,z:1},camera:clone(DEFAULT_CAMERA),dragmode:'orbit',
                 bgcolor:'#ffffff',domain:{x:[0,1],y:[0,1]}}};
}
function cameraOf(gd) {
  const c=gd._fullLayout.scene.camera;
  return {eye:{x:c.eye.x,y:c.eye.y,z:c.eye.z},up:{x:c.up.x,y:c.up.y,z:c.up.z},
          center:{x:c.center.x,y:c.center.y,z:c.center.z},projection:{type:c.projection.type}};
}
function exportSize(gd) {
  const width=Math.round(gd._fullLayout.width),height=Math.round(gd._fullLayout.height);
  const scale=Math.max(3,2400/Math.max(width,height));
  return {format:'png',width,height,scale,pixel_width:Math.round(width*scale),pixel_height:Math.round(height*scale)};
}
function viewRecord(domain) {
  const gd=plots[domain], f=gd._fullLayout, s=f.scene, source=DATA.domains[domain];
  return {camera:cameraOf(gd),n_samples:source.sample_ids.length,
          coordinates:source.coordinates, sample_ids:source.sample_ids,
          viewport:{width:f.width,height:f.height,margin:{l:f.margin.l,r:f.margin.r,t:f.margin.t,b:f.margin.b,pad:f.margin.pad},
                    scene:{domain:clone(s.domain),aspectmode:s.aspectmode,aspectratio:clone(s.aspectratio),
                           ranges:{x:clone(s.xaxis.range),y:clone(s.yaxis.range),z:clone(s.zaxis.range)}}},
          png_export:exportSize(gd)};
}
function currentViews() {
  return {schema:'bacteria-analysis.hmds-3d-camera.v1',exported_at:new Date().toISOString(),
          viewer:'offline Plotly camera viewer',plotly_version:Plotly.version,
          operation:'camera changes only; coordinates, fits and point colors unchanged',
          coordinate_system:'native saved Poincare ball x,y,z; no transformation',
          color_reference:DATA.color_reference,marker:DATA.marker,
          views:Object.fromEntries(DOMAINS.map(d=>[d,viewRecord(d)]))};
}
function refreshJSON() {
  if (ready) textarea.value=JSON.stringify(currentViews(),null,2);
}
function downloadJSON() {
  refreshJSON();
  const blob=new Blob([textarea.value+'\n'],{type:'application/json'});
  const url=URL.createObjectURL(blob),link=document.createElement('a');
  link.href=url;link.download='hmds_3d_views.json';document.body.appendChild(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
  status.textContent='Views JSON downloaded.';
}
async function copyJSON() {
  refreshJSON();
  try { await navigator.clipboard.writeText(textarea.value); }
  catch (_) { textarea.focus();textarea.select();
    if (!document.execCommand('copy')) {status.textContent='JSON selected. Press Ctrl/Cmd+C to copy.';return;} }
  status.textContent='Views JSON copied.';
}
async function savePNG(domain) {
  status.textContent=`Exporting ${domain} PNG…`;
  const gd=plots[domain],size=exportSize(gd);
  try {await Plotly.downloadImage(gd,{format:size.format,width:size.width,height:size.height,
                                    scale:size.scale,filename:`hmds_3d_${domain}`});
       status.textContent=`${domain[0].toUpperCase()+domain.slice(1)} PNG downloaded.`;}
  catch (error) {status.textContent=`PNG export failed: ${error.message}`;}
  refreshJSON();
}
async function init() {
  await Promise.all(DOMAINS.map(domain=>{
    const source=DATA.domains[domain];
    const points={type:'scatter3d',mode:'markers',x:source.x,y:source.y,z:source.z,
      text:source.sample_ids,hovertemplate:'%{text}<extra></extra>',showlegend:false,
      marker:{size:DATA.marker.size,color:source.colors,opacity:1,line:{color:'#ffffff',width:.5}}};
    return Plotly.newPlot(plots[domain],[wireframe(),points],layout(),
      {responsive:true,scrollZoom:true,displayModeBar:false,displaylogo:false});
  }));
  ready=true;
  DOMAINS.forEach(domain=>{
    plots[domain].on('plotly_relayout',()=>requestAnimationFrame(refreshJSON));
    document.getElementById(`download-${domain}-png`).addEventListener('click',()=>savePNG(domain));
  });
  document.getElementById('reset-views').addEventListener('click',async()=>{
    await Promise.all(DOMAINS.map(d=>Plotly.relayout(plots[d],{'scene.camera':clone(DEFAULT_CAMERA)})));
    refreshJSON();status.textContent='Both cameras reset.';
  });
  document.getElementById('download-views').addEventListener('click',downloadJSON);
  document.getElementById('copy-views').addEventListener('click',copyJSON);
  document.querySelectorAll('button').forEach(b=>b.disabled=false);
  window.addEventListener('resize',()=>setTimeout(refreshJSON,150));
  refreshJSON();status.textContent='Ready · 106 samples per view · works offline';
}
init().catch(error=>{status.textContent=`Viewer could not load: ${error.message}`;console.error(error);});
</script>
</body></html>
'''


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_viewer(output_dir):
    """Validate saved coordinates/colors and write one fully self-contained HTML."""
    out = Path(output_dir).resolve()
    repo = out.parent.parent
    color_path = repo / "output/jupyter-notebook/chemical_hmds_20260928_230847_298095/color_reference/aid_to_chemical_color.csv"
    color_frame = pd.read_csv(color_path, index_col=0)
    if not color_frame.index.is_unique or len(color_frame) != 106:
        raise ValueError("Expected 106 unique frozen notebook colors")
    data = dict(domains={}, marker=dict(size=6, opacity=1, edge_color="#ffffff", edge_width=.5),
                color_reference=dict(path=str(color_path), sha256=_hash(color_path),
                                     column="color_hex", mapping=color_frame.color_hex.to_dict()),
                bundled_plotly_python_version=plotly.__version__)
    hashes = {color_path: _hash(color_path)}
    ids = None
    for domain in ("neural", "chemical"):
        path = out / f"hmds_full106/3d/{domain}/coordinates.csv"
        frame = pd.read_csv(path, index_col=0)
        if not frame.index.is_unique or len(frame) != 106 or set(frame.index) != set(color_frame.index):
            raise ValueError(f"{domain} coordinates must match all 106 frozen sample IDs")
        if ids is None:
            ids = frame.index
        frame = frame.loc[ids, ["x", "y", "z"]]
        xyz = frame.to_numpy(float)
        if not np.isfinite(xyz).all() or np.any(np.linalg.norm(xyz, axis=1) > 1 + 1e-12):
            raise ValueError(f"{domain} contains invalid Poincare ball coordinates")
        colors = color_frame.loc[ids, "color_hex"].tolist()
        if not all(re.fullmatch(r"#[0-9a-fA-F]{6}", color) for color in colors):
            raise ValueError("Frozen point colors must be six-digit hexadecimal values")
        hashes[path] = _hash(path)
        data["domains"][domain] = dict(sample_ids=ids.tolist(), colors=colors,
                                      coordinates=dict(path=str(path), sha256=hashes[path]),
                                      **{axis: frame[axis].tolist() for axis in ("x", "y", "z")})
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    html = HTML.replace("__VIEWER_DATA__", payload)
    html = html.replace("__PLOTLY_JS__", get_plotlyjs().replace("</script", "<\\/script"))
    destination = out / "hmds_3d_viewer.html"
    destination.write_text(html, encoding="utf-8")
    if any(_hash(path) != digest for path, digest in hashes.items()):
        raise ValueError("A saved scientific input changed during viewer export")
    return dict(path=str(destination), size_bytes=destination.stat().st_size,
                n_samples=106, coordinate_hashes={d: v["coordinates"]["sha256"] for d, v in data["domains"].items()},
                offline_plotly_bundled=True, input_hashes_unchanged=True)


if __name__ == "__main__":
    print(json.dumps(export_viewer(Path(__file__).resolve().parents[1]), indent=2))
