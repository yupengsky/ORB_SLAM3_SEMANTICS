#!/usr/bin/env python3
import argparse
import http.server
import socketserver
from pathlib import Path


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLAM Point Cloud Replay</title>
<style>
html, body { margin: 0; height: 100%; background: #111; color: #eee; font-family: system-ui, sans-serif; }
#canvas { width: 100vw; height: 100vh; display: block; cursor: grab; }
#panel { position: fixed; left: 16px; right: 16px; bottom: 14px; display: grid; grid-template-columns: auto 1fr auto auto; gap: 12px; align-items: center; background: rgba(20,20,20,.82); border: 1px solid #444; padding: 10px 12px; }
button { height: 34px; padding: 0 14px; border: 1px solid #777; background: #222; color: #fff; }
input[type=range] { width: 100%; }
#stats { font-variant-numeric: tabular-nums; white-space: nowrap; }
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div id="panel">
  <button id="play">Play</button>
  <input id="slider" type="range" min="0" max="0" value="0">
  <span id="stats">loading</span>
  <button id="reset">Reset</button>
</div>
<script>
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const slider = document.getElementById("slider");
const play = document.getElementById("play");
const reset = document.getElementById("reset");
const stats = document.getElementById("stats");
let points = [];
let frame = 0;
let maxFrame = 0;
let playing = false;
let scale = 230;
let yaw = -0.6;
let pitch = 0.45;
let dragging = false;
let last = [0, 0];

function resize() {
  canvas.width = Math.floor(innerWidth * devicePixelRatio);
  canvas.height = Math.floor(innerHeight * devicePixelRatio);
  draw();
}

function parsePly(text) {
  const lines = text.split(/\r?\n/);
  let vertexCount = 0;
  let headerEnd = 0;
  let props = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith("element vertex ")) vertexCount = Number(line.split(/\s+/)[2]);
    if (line.startsWith("property ")) props.push(line.split(/\s+/).pop());
    if (line === "end_header") { headerEnd = i + 1; break; }
  }
  const idx = Object.fromEntries(props.map((p, i) => [p, i]));
  const parsed = [];
  for (let i = 0; i < vertexCount; i++) {
    const values = lines[headerEnd + i].trim().split(/\s+/).map(Number);
    if (values.length < props.length) continue;
    const p = {
      x: values[idx.x], y: values[idx.y], z: values[idx.z],
      r: values[idx.red] ?? 190, g: values[idx.green] ?? 190, b: values[idx.blue] ?? 190,
      first: values[idx.first_seen_frame_index] ?? 0
    };
    maxFrame = Math.max(maxFrame, p.first);
    parsed.push(p);
  }
  return parsed;
}

function project(p) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * p.x + sy * p.z;
  const z1 = -sy * p.x + cy * p.z;
  const y1 = cp * p.y - sp * z1;
  return [canvas.width / 2 + x1 * scale * devicePixelRatio, canvas.height / 2 - y1 * scale * devicePixelRatio];
}

function draw() {
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  let visible = 0;
  for (const p of points) {
    if (p.first > frame) continue;
    visible++;
    const [x, y] = project(p);
    ctx.fillStyle = `rgb(${p.r},${p.g},${p.b})`;
    ctx.fillRect(x, y, 2 * devicePixelRatio, 2 * devicePixelRatio);
  }
  stats.textContent = `${frame}/${maxFrame}  points ${visible}/${points.length}`;
}

function tick() {
  if (playing) {
    frame = frame >= maxFrame ? 0 : frame + 1;
    slider.value = frame;
    draw();
  }
  requestAnimationFrame(tick);
}

play.onclick = () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; };
reset.onclick = () => { frame = 0; slider.value = 0; yaw = -0.6; pitch = 0.45; scale = 230; draw(); };
slider.oninput = () => { frame = Number(slider.value); draw(); };
canvas.onmousedown = e => { dragging = true; last = [e.clientX, e.clientY]; };
canvas.onmouseup = () => dragging = false;
canvas.onmouseleave = () => dragging = false;
canvas.onmousemove = e => {
  if (!dragging) return;
  yaw += (e.clientX - last[0]) * 0.01;
  pitch += (e.clientY - last[1]) * 0.01;
  last = [e.clientX, e.clientY];
  draw();
};
canvas.onwheel = e => { e.preventDefault(); scale *= e.deltaY > 0 ? 0.9 : 1.1; draw(); };

fetch("/cloud.ply").then(r => r.text()).then(text => {
  points = parsePly(text);
  slider.max = maxFrame;
  stats.textContent = "ready";
  resize();
  tick();
});
addEventListener("resize", resize);
</script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Replay an ORB-SLAM3 point cloud timeline PLY in a browser.")
    parser.add_argument("ply")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    ply = Path(args.ply).expanduser().resolve()
    if not ply.exists():
        raise FileNotFoundError(ply)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/cloud.ply":
                data = ply.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            return

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer((args.host, args.port), Handler) as server:
        print(f"http://{args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
