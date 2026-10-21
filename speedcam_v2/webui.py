"""Flask web UI that mirrors the older RadarPi workflow."""

from __future__ import annotations

import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import cv2
from flask import Flask, Response, jsonify, redirect, render_template_string, request

from .config import DEFAULT_CONFIG, save_config


HTML_TEMPLATE = """
<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ app_name }}</title>
  <style>
    html, body { background:#111; color:#eee; font-family:Arial,sans-serif; margin:0; padding:0; min-height:100vh; }
    body { text-align:center; }
    .shell { padding:8px 8px 10px; }
    .shell:fullscreen, .shell:-webkit-full-screen {
      width:100vw; height:100vh; padding:0; box-sizing:border-box;
      display:flex; flex-direction:column; gap:0; background:#111;
      position:relative; overflow:hidden;
    }
    img {
      display:block;
      width:100%;
      max-width:none;
      border:1px solid #333;
      border-radius:8px;
      margin:0 auto;
      object-fit:contain;
      max-height:calc(100vh - 74px);
      background:#000;
    }
    .row { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin:0 0 8px; }
    .row form { margin:0; }
    .stack { display:flex; flex-direction:column; gap:8px; align-items:center; }
    button {
      background:#2d3f75; color:#fff; border:1px solid #9fb4ff; border-radius:6px;
      padding:8px 12px; min-width:84px; cursor:pointer; font-size:15px;
    }
    button.active { background:#c8aa2b; color:#111; border-color:#fff2b0; }
    button.warn { background:#5c2d2d; border-color:#ffcaca; }
    button.dim { background:#2d2d2d; border-color:#777; }
    .panel {
      background:#1a1a1a; border:1px solid #444; border-radius:8px; padding:10px; margin:0 0 8px;
      max-width:960px; margin-left:auto; margin-right:auto;
    }
    .panel h3 { margin:0 0 8px; font-size:16px; }
    .panel p { margin:0 0 8px; color:#bbb; font-size:13px; }
    .panel input {
      background:#0f0f0f; color:#fff; border:1px solid #666; border-radius:6px;
      padding:8px 10px; font-size:16px; width:140px; text-align:center;
    }
    .viewer { position:relative; width:100%; max-width:960px; margin:0 auto; }
    .shell:fullscreen .viewer, .shell:-webkit-full-screen .viewer {
      flex:1 1 auto;
      width:100%;
      height:100vh;
      max-width:none;
      display:flex;
      align-items:center;
      justify-content:center;
      margin:0;
    }
    .shell:fullscreen img, .shell:-webkit-full-screen img {
      width:auto;
      height:100%;
      max-width:100%;
      max-height:100%;
      object-fit:contain;
    }
    .shell:fullscreen > .row:first-child, .shell:-webkit-full-screen > .row:first-child {
      position:absolute;
      top:8px;
      left:8px;
      right:8px;
      z-index:20;
      margin:0;
      pointer-events:none;
    }
    .shell:fullscreen > .row:first-child form, .shell:-webkit-full-screen > .row:first-child form,
    .shell:fullscreen > .row:first-child button, .shell:-webkit-full-screen > .row:first-child button {
      pointer-events:auto;
    }
    .shell:fullscreen .panel, .shell:-webkit-full-screen .panel,
    .shell:fullscreen > .row:nth-child(2), .shell:-webkit-full-screen > .row:nth-child(2) {
      position:absolute;
      top:58px;
      left:8px;
      right:8px;
      z-index:20;
      margin:0 auto;
      max-width:960px;
    }
    .viewer canvas {
      position:absolute; inset:0; width:100%; height:100%; touch-action:none;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="row">
      <form method="post" action="/action/SOL"><button class="{{ 'active' if profile == 'SOL' else '' }}">SOL</button></form>
      <form method="post" action="/action/SAG"><button class="{{ 'active' if profile == 'SAG' else '' }}">SAG</button></form>
      <form method="post" action="/action/MENU"><button class="{{ 'active' if menu_open else '' }}">MENU</button></form>
      <button type="button" id="fullscreenBtn" class="dim">Tam Ekran</button>
    </div>
    {% if menu_open and active_tool == 'none' %}
    <div class="row">
      <form method="post" action="/action/SARI"><button>Sari Cizgi</button></form>
      <form method="post" action="/action/KIRMIZI"><button>Kirmizi Cizgi</button></form>
      <form method="post" action="/action/KALIB"><button>Kalibrasyon</button></form>
      <form method="post" action="/action/MESAFE"><button>Mesafe Gir</button></form>
      <form method="post" action="/action/LIMIT"><button>Hiz Limiti: {{ speed_limit_kmh|round(0)|int }}</button></form>
      <form method="post" action="/action/TESPITMODE"><button>Tespit: {{ detection_mode_title }}</button></form>
      <form method="post" action="/action/PERFMODE"><button>Performans: {{ performance_mode_title }}</button></form>
      <form method="post" action="/action/ROI"><button>{{ 'Tespit Alani Kapat' if show_roi_overlay else 'Tespit Alani Ac' }}</button></form>
      <form method="post" action="/action/VARSAY"><button>Varsayilan</button></form>
      <form method="post" action="/action/CIK"><button class="dim">Kapat</button></form>
    </div>
    {% endif %}
    {% if active_tool == 'manual_distance' %}
    <div class="panel">
      <h3>Mesafe Gir</h3>
      <p>Sari ve kirmizi cizgi arasi gercek mesafeyi metre olarak girin.</p>
      <form class="stack" method="post" action="/set_distance">
        <input type="number" step="0.1" min="0.1" name="distance" value="{{ distance_m }}">
        <div class="row">
          <button type="submit">Kaydet</button>
          <button type="submit" formaction="/action/CANCELWEB" formmethod="post" class="dim">Iptal</button>
        </div>
      </form>
    </div>
    {% elif active_tool == 'manual_speed_limit' %}
    <div class="panel">
      <h3>Hiz Limiti Gir</h3>
      <p>Limit degerini km/h olarak girin.</p>
      <form class="stack" method="post" action="/set_speed_limit">
        <input type="number" step="0.1" min="1" name="speed_limit" value="{{ speed_limit_kmh }}">
        <div class="row">
          <button type="submit">Kaydet</button>
          <button type="submit" formaction="/action/CANCELWEB" formmethod="post" class="dim">Iptal</button>
        </div>
      </form>
    </div>
    {% elif active_tool in ['edit_start', 'edit_finish'] %}
    <div class="panel">
      <h3>{{ 'Sari Cizgi Ayarla' if active_tool == 'edit_start' else 'Kirmizi Cizgi Ayarla' }}</h3>
      <p>Uclari veya cizginin govdesini parmaginizla surukleyin. Sonra Kaydet'e basin.</p>
      <div class="row">
        <form method="post" action="/action/SAVEWEB"><button>Kaydet</button></form>
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
    </div>
    {% elif active_tool == 'calibration' %}
    <div class="panel">
      <h3>Kalibrasyon</h3>
      <p>Mavi tespit alaninin koselerini surukleyin. Sonra Kaydet'e basin.</p>
      <div class="row">
        <form method="post" action="/action/SAVEWEB"><button>Kaydet</button></form>
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
    </div>
    {% elif active_tool == 'detection_mode' %}
    <div class="panel">
      <h3>Tespit Modu</h3>
      <p>Yakalama karakterini secin. Hassas daha erken yakalar, Kararli daha secicidir.</p>
      <div class="row">
        <form method="post" action="/action/KARARLI"><button class="{{ 'active' if detection_mode == 'KARARLI' else '' }}">Kararli</button></form>
        <form method="post" action="/action/DENGELI"><button class="{{ 'active' if detection_mode == 'DENGELI' else '' }}">Dengeli</button></form>
        <form method="post" action="/action/HASSAS"><button class="{{ 'active' if detection_mode == 'HASSAS' else '' }}">Hassas</button></form>
      </div>
      <div class="row">
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
    </div>
    {% elif active_tool == 'performance_mode' %}
    <div class="panel">
      <h3>Performans Modu</h3>
      <p>Normal daha dengeli, Hizli daha hafif akistir.</p>
      <div class="row">
        <form method="post" action="/action/NORMAL"><button class="{{ 'active' if performance_mode == 'NORMAL' else '' }}">Normal</button></form>
        <form method="post" action="/action/HIZLI"><button class="{{ 'active' if performance_mode == 'HIZLI' else '' }}">Hizli</button></form>
      </div>
      <div class="row">
        <form method="post" action="/action/CANCELWEB"><button class="dim">Iptal</button></form>
      </div>
    </div>
    {% endif %}
    <div class="viewer">
      <img id="stream" src="/video_feed" alt="RadarPi stream">
      {% if active_tool in ['edit_start', 'edit_finish', 'calibration'] %}
      <canvas id="editorCanvas"></canvas>
      {% endif %}
    </div>
  </div>
  <script>
    (function () {
      const shell = document.querySelector(".shell");
      const btn = document.getElementById("fullscreenBtn");
      if (!shell || !btn) return;

      async function toggleFullscreen() {
        try {
          const active = document.fullscreenElement || document.webkitFullscreenElement;
          if (active) {
            if (document.exitFullscreen) await document.exitFullscreen();
            else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
            return;
          }
          if (shell.requestFullscreen) await shell.requestFullscreen();
          else if (shell.webkitRequestFullscreen) shell.webkitRequestFullscreen();
        } catch (err) {
          console.warn("Tam ekran acilamadi:", err);
        }
      }

      function syncLabel() {
        const active = document.fullscreenElement || document.webkitFullscreenElement;
        btn.textContent = active ? "Kapat" : "Tam Ekran";
      }

      btn.addEventListener("click", toggleFullscreen);
      document.addEventListener("fullscreenchange", syncLabel);
      document.addEventListener("webkitfullscreenchange", syncLabel);
      syncLabel();
    })();
  </script>
  {% if active_tool in ['edit_start', 'edit_finish', 'calibration'] %}
  <script>
    const FRAME_W = {{ frame_width }};
    const FRAME_H = {{ frame_height }};
    const ACTIVE_TOOL = "{{ active_tool }}";
    const img = document.getElementById("stream");
    const canvas = document.getElementById("editorCanvas");
    const ctx = canvas.getContext("2d");
    let layout = null;
    let drag = null;
    let lastSent = 0;
    let imageBox = {x: 0, y: 0, width: 1, height: 1};

    function resizeCanvas() {
      const rect = img.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      const frameAspect = FRAME_W / FRAME_H;
      const rectAspect = rect.width / Math.max(rect.height, 1);
      let drawWidth = rect.width;
      let drawHeight = rect.height;
      if (rectAspect > frameAspect) {
        drawHeight = rect.height;
        drawWidth = drawHeight * frameAspect;
      } else {
        drawWidth = rect.width;
        drawHeight = drawWidth / frameAspect;
      }
      imageBox = {
        x: (rect.width - drawWidth) / 2,
        y: (rect.height - drawHeight) / 2,
        width: drawWidth,
        height: drawHeight,
      };
      draw();
    }

    function toCanvas(pt) {
      return {
        x: imageBox.x + pt[0] * (imageBox.width / FRAME_W),
        y: imageBox.y + pt[1] * (imageBox.height / FRAME_H),
      };
    }

    function toFrame(x, y) {
      const clampedX = Math.max(imageBox.x, Math.min(imageBox.x + imageBox.width, x));
      const clampedY = Math.max(imageBox.y, Math.min(imageBox.y + imageBox.height, y));
      return {
        x: Math.max(0, Math.min(FRAME_W - 1, Math.round((clampedX - imageBox.x) * (FRAME_W / imageBox.width)))),
        y: Math.max(0, Math.min(FRAME_H - 1, Math.round((clampedY - imageBox.y) * (FRAME_H / imageBox.height)))),
      };
    }

    function near(a, b, r=18) {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      return (dx*dx + dy*dy) <= r*r;
    }

    function pointNearSegment(p, a, b, tolerance=16) {
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      if (dx === 0 && dy === 0) return near(p, a, tolerance);
      let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / ((dx*dx) + (dy*dy));
      t = Math.max(0, Math.min(1, t));
      const proj = {x: a.x + t*dx, y: a.y + t*dy};
      return near(p, proj, tolerance);
    }

    async function loadLayout() {
      const res = await fetch("/api/layout");
      layout = await res.json();
      draw();
    }

    function draw() {
      if (!layout) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const start1 = toCanvas(layout.start_p1);
      const start2 = toCanvas(layout.start_p2);
      const finish1 = toCanvas(layout.finish_p1);
      const finish2 = toCanvas(layout.finish_p2);
      const lane = layout.lane_polygon.map(toCanvas);

      ctx.lineWidth = 2;
      ctx.strokeStyle = "#2f7dff";
      ctx.beginPath();
      ctx.moveTo(lane[0].x, lane[0].y);
      for (let i = 1; i < lane.length; i++) ctx.lineTo(lane[i].x, lane[i].y);
      ctx.closePath();
      ctx.stroke();

      ctx.lineWidth = 3;
      ctx.strokeStyle = "#ffe600";
      ctx.beginPath(); ctx.moveTo(start1.x, start1.y); ctx.lineTo(start2.x, start2.y); ctx.stroke();
      ctx.strokeStyle = "#ff4040";
      ctx.beginPath(); ctx.moveTo(finish1.x, finish1.y); ctx.lineTo(finish2.x, finish2.y); ctx.stroke();

      if (ACTIVE_TOOL === "edit_start") {
        for (const p of [start1, start2]) {
          ctx.fillStyle = "#ffffff";
          ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = "#ffe600";
          ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
        }
      } else if (ACTIVE_TOOL === "edit_finish") {
        for (const p of [finish1, finish2]) {
          ctx.fillStyle = "#ffffff";
          ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = "#ff4040";
          ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
        }
      } else if (ACTIVE_TOOL === "calibration") {
        for (const p of lane) {
          ctx.fillStyle = "#ffffff";
          ctx.beginPath(); ctx.arc(p.x, p.y, 9, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = "#2f7dff";
          ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
        }
      }
    }

    async function pushLayout() {
      const now = Date.now();
      if ((now - lastSent) < 50) return;
      lastSent = now;
      await fetch("/api/layout", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(layout)
      });
    }

    function pointerDown(ev) {
      if (!layout) return;
      const rect = canvas.getBoundingClientRect();
      const p = {x: ev.clientX - rect.left, y: ev.clientY - rect.top};
      const start1 = toCanvas(layout.start_p1);
      const start2 = toCanvas(layout.start_p2);
      const finish1 = toCanvas(layout.finish_p1);
      const finish2 = toCanvas(layout.finish_p2);
      const lane = layout.lane_polygon.map(toCanvas);

      if (ACTIVE_TOOL === "edit_start") {
        if (near(p, start1)) drag = {kind: "start_p1"};
        else if (near(p, start2)) drag = {kind: "start_p2"};
        else if (pointNearSegment(p, start1, start2)) drag = {kind: "start_line", last: toFrame(p.x, p.y)};
      } else if (ACTIVE_TOOL === "edit_finish") {
        if (near(p, finish1)) drag = {kind: "finish_p1"};
        else if (near(p, finish2)) drag = {kind: "finish_p2"};
        else if (pointNearSegment(p, finish1, finish2)) drag = {kind: "finish_line", last: toFrame(p.x, p.y)};
      } else if (ACTIVE_TOOL === "calibration") {
        lane.forEach((pt, index) => {
          if (!drag && near(p, pt)) drag = {kind: "lane_point", index: index};
        });
      }
      if (drag) ev.preventDefault();
    }

    function pointerMove(ev) {
      if (!drag || !layout) return;
      const rect = canvas.getBoundingClientRect();
      const p = toFrame(ev.clientX - rect.left, ev.clientY - rect.top);
      if (drag.kind === "start_p1") layout.start_p1 = [p.x, p.y];
      else if (drag.kind === "start_p2") layout.start_p2 = [p.x, p.y];
      else if (drag.kind === "finish_p1") layout.finish_p1 = [p.x, p.y];
      else if (drag.kind === "finish_p2") layout.finish_p2 = [p.x, p.y];
      else if (drag.kind === "lane_point") layout.lane_polygon[drag.index] = [p.x, p.y];
      else if (drag.kind === "start_line") {
        const dx = p.x - drag.last.x;
        const dy = p.y - drag.last.y;
        layout.start_p1 = [layout.start_p1[0] + dx, layout.start_p1[1] + dy];
        layout.start_p2 = [layout.start_p2[0] + dx, layout.start_p2[1] + dy];
        drag.last = p;
      } else if (drag.kind === "finish_line") {
        const dx = p.x - drag.last.x;
        const dy = p.y - drag.last.y;
        layout.finish_p1 = [layout.finish_p1[0] + dx, layout.finish_p1[1] + dy];
        layout.finish_p2 = [layout.finish_p2[0] + dx, layout.finish_p2[1] + dy];
        drag.last = p;
      }
      draw();
      pushLayout();
    }

    function pointerUp() {
      drag = null;
    }

    window.addEventListener("resize", resizeCanvas);
    img.addEventListener("load", resizeCanvas);
    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointermove", pointerMove);
    window.addEventListener("pointerup", pointerUp);
    loadLayout();
  </script>
  {% endif %}
</body>
</html>
"""


DETECTION_MODE_TITLES = {
    "KARARLI": "Kararli",
    "DENGELI": "Dengeli",
    "HASSAS": "Hassas",
}

PERFORMANCE_MODE_TITLES = {
    "NORMAL": "Normal",
    "HIZLI": "Hizli",
}


@dataclass
class WebState:
    """Shared state for the Flask app."""

    latest_jpeg: bytes = b""
    latest_status: dict[str, Any] = field(default_factory=dict)
    menu_open: bool = False
    control_mode: str = "NONE"
    active_tool: str = "none"
    lock: threading.Lock = field(default_factory=threading.Lock)


def _current_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Return mutable active profile dict."""
    return config["profiles"][config["app"]["active_profile"]]


def _set_tool(web_state: WebState, tool: str) -> None:
    """Update current tool under lock."""
    with web_state.lock:
        web_state.menu_open = True
        web_state.active_tool = tool
        web_state.control_mode = tool.upper()


def _close_tool(web_state: WebState) -> None:
    """Reset tool state while keeping menu open."""
    with web_state.lock:
        web_state.menu_open = True
        web_state.active_tool = "none"
        web_state.control_mode = "NONE"


def _line_move(line: list[list[float]], delta_y: int) -> list[list[float]]:
    """Move line by integer y delta."""
    return [[int(point[0]), int(point[1] + delta_y)] for point in line]


def _reset_active_profile(config: dict[str, Any]) -> None:
    """Reset current profile and UI modes back to defaults."""
    active_profile = config["app"]["active_profile"]
    config["profiles"][active_profile] = deepcopy(DEFAULT_CONFIG["profiles"][active_profile])
    config["app"]["show_roi_overlay"] = bool(DEFAULT_CONFIG["app"]["show_roi_overlay"])
    config["app"]["detection_mode"] = str(DEFAULT_CONFIG["app"]["detection_mode"])
    config["app"]["performance_mode"] = str(DEFAULT_CONFIG["app"]["performance_mode"])


def create_app(config: dict, web_state: WebState) -> Flask:
    """Create Flask app with RadarPi-compatible workflow."""
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        with web_state.lock:
            status = dict(web_state.latest_status)
            menu_open = web_state.menu_open
            control_mode = web_state.control_mode
            active_tool = web_state.active_tool
        return render_template_string(
            HTML_TEMPLATE,
            app_name=config["app"]["name"],
            profile=config["app"]["active_profile"],
            status=status or {
                "status_text": "Bekleniyor",
                "fps": 0.0,
                "active_tracks": 0,
                "last_speed_text": "--",
                "backend_name": "--",
            },
            menu_open=menu_open,
            control_mode=control_mode,
            active_tool=active_tool,
            draw_guides=bool(config["app"].get("draw_guides", True)),
            show_roi_overlay=bool(config["app"].get("show_roi_overlay", True)),
            detection_mode=str(config["app"].get("detection_mode", "DENGELI")).upper(),
            performance_mode=str(config["app"].get("performance_mode", "NORMAL")).upper(),
            detection_mode_title=DETECTION_MODE_TITLES.get(str(config["app"].get("detection_mode", "DENGELI")).upper(), "Dengeli"),
            performance_mode_title=PERFORMANCE_MODE_TITLES.get(str(config["app"].get("performance_mode", "NORMAL")).upper(), "Normal"),
            distance_m=float(config["measurement"]["real_distance_meters"]),
            speed_limit_kmh=float(config["measurement"]["speed_limit_kmh"]),
            frame_width=int(config["camera"]["width"]),
            frame_height=int(config["camera"]["height"]),
        )

    @app.post("/action/<action>")
    def action(action: str) -> Any:
        action_name = action.upper()
        profile = _current_profile(config)

        if action_name in {"SOL", "SAG"}:
            config["app"]["active_profile"] = action_name
            _close_tool(web_state)
        elif action_name == "MENU":
            with web_state.lock:
                web_state.menu_open = not web_state.menu_open
                if not web_state.menu_open:
                    web_state.active_tool = "none"
                    web_state.control_mode = "NONE"
        elif action_name == "CIK":
            with web_state.lock:
                web_state.menu_open = False
                web_state.active_tool = "none"
                web_state.control_mode = "NONE"
        elif action_name == "SARI":
            _set_tool(web_state, "edit_start")
        elif action_name == "KIRMIZI":
            _set_tool(web_state, "edit_finish")
        elif action_name == "KALIB":
            _set_tool(web_state, "calibration")
        elif action_name == "MESAFE":
            _set_tool(web_state, "manual_distance")
        elif action_name == "LIMIT":
            _set_tool(web_state, "manual_speed_limit")
        elif action_name == "TESPITMODE":
            _set_tool(web_state, "detection_mode")
        elif action_name == "PERFMODE":
            _set_tool(web_state, "performance_mode")
        elif action_name == "KARARLI":
            config["app"]["detection_mode"] = "KARARLI"
            _close_tool(web_state)
        elif action_name == "DENGELI":
            config["app"]["detection_mode"] = "DENGELI"
            _close_tool(web_state)
        elif action_name == "HASSAS":
            config["app"]["detection_mode"] = "HASSAS"
            _close_tool(web_state)
        elif action_name == "NORMAL":
            config["app"]["performance_mode"] = "NORMAL"
            config["performance"]["jpeg_quality"] = 70
            config["performance"]["queue_size"] = 1
            config["camera"]["fps"] = 25
            _close_tool(web_state)
        elif action_name == "HIZLI":
            config["app"]["performance_mode"] = "HIZLI"
            config["performance"]["jpeg_quality"] = 60
            config["performance"]["queue_size"] = 1
            config["camera"]["fps"] = 20
            _close_tool(web_state)
        elif action_name == "ROI":
            config["app"]["show_roi_overlay"] = not bool(config["app"].get("show_roi_overlay", True))
            _close_tool(web_state)
        elif action_name == "VARSAY":
            _reset_active_profile(config)
            _close_tool(web_state)
        elif action_name == "SAVEWEB":
            _close_tool(web_state)
        elif action_name == "CANCELWEB":
            _close_tool(web_state)
        elif action_name == "SARI_UP":
            profile["yellow_line"] = _line_move(profile["yellow_line"], -int(config["app"]["line_move_step_px"]))
        elif action_name == "SARI_DOWN":
            profile["yellow_line"] = _line_move(profile["yellow_line"], int(config["app"]["line_move_step_px"]))
        elif action_name == "KIRMIZI_UP":
            profile["red_line"] = _line_move(profile["red_line"], -int(config["app"]["line_move_step_px"]))
        elif action_name == "KIRMIZI_DOWN":
            profile["red_line"] = _line_move(profile["red_line"], int(config["app"]["line_move_step_px"]))

        save_config(config)
        return redirect("/")

    @app.post("/set_distance")
    def set_distance() -> Any:
        try:
            value = max(0.1, float(request.form.get("distance", "10.0")))
            config["measurement"]["real_distance_meters"] = round(value, 2)
            save_config(config)
        finally:
            _close_tool(web_state)
        return redirect("/")

    @app.post("/set_speed_limit")
    def set_speed_limit() -> Any:
        try:
            value = max(1.0, float(request.form.get("speed_limit", "50.0")))
            config["measurement"]["speed_limit_kmh"] = round(value, 1)
            save_config(config)
        finally:
            _close_tool(web_state)
        return redirect("/")

    @app.get("/api/status")
    def status() -> Response:
        with web_state.lock:
            payload = dict(web_state.latest_status)
        return jsonify(payload)

    @app.get("/api/config")
    def get_config() -> Response:
        return jsonify(config)

    @app.get("/api/layout")
    def get_layout() -> Response:
        profile = _current_profile(config)
        return jsonify(
            {
                "start_p1": profile["yellow_line"][0],
                "start_p2": profile["yellow_line"][1],
                "finish_p1": profile["red_line"][0],
                "finish_p2": profile["red_line"][1],
                "lane_polygon": profile["lane_polygon"],
            }
        )

    @app.post("/api/layout")
    def update_layout() -> Response:
        payload = request.get_json(force=True, silent=True) or {}
        profile = _current_profile(config)
        if "start_p1" in payload and "start_p2" in payload:
            profile["yellow_line"] = [payload["start_p1"], payload["start_p2"]]
        if "finish_p1" in payload and "finish_p2" in payload:
            profile["red_line"] = [payload["finish_p1"], payload["finish_p2"]]
        if "lane_polygon" in payload and isinstance(payload["lane_polygon"], list) and len(payload["lane_polygon"]) == 4:
            profile["lane_polygon"] = payload["lane_polygon"]
        save_config(config)
        return jsonify({"ok": True})

    @app.get("/video_feed")
    def video_feed() -> Response:
        def generate() -> Any:
            while True:
                with web_state.lock:
                    frame = web_state.latest_jpeg
                if frame:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    )

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def encode_frame(frame, quality: int = 85) -> bytes:
    """JPEG-encode frame for web streaming."""
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return encoded.tobytes() if ok else b""
