"""Hoop calibration by clicking on a frame inside a Colab notebook (3 clicks -> hoop.json dict)."""

import base64

_JS = """
async function pickHoop(dataUrl, width, height) {
  const labels = ['LEFT rim edge', 'RIGHT rim edge', 'FRONT rim edge (nearest to camera)'];
  const colours = ['#e33', '#3c3', '#08f'];
  const div = document.createElement('div');
  const info = document.createElement('div');
  info.style.cssText = 'font: 16px sans-serif; margin: 6px 0;';
  const reset = document.createElement('button');
  reset.textContent = 'Reset clicks';
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  canvas.style.cssText = 'max-width: 100%; cursor: crosshair; display: block;';
  div.append(info, reset, canvas);
  document.body.appendChild(div);
  const ctx = canvas.getContext('2d');
  const img = new Image();
  await new Promise(r => { img.onload = r; img.src = dataUrl; });
  let pts = [];
  const r = Math.max(4, width / 300);
  function redraw() {
    ctx.drawImage(img, 0, 0);
    pts.forEach((p, k) => {
      ctx.fillStyle = colours[k];
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI); ctx.fill();
    });
    info.textContent = pts.length < 3 ? `Click ${pts.length + 1}/3: ${labels[pts.length]}` : 'Done';
  }
  redraw();
  google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
  return await new Promise(resolve => {
    reset.onclick = () => { pts = []; redraw(); };
    canvas.onclick = e => {
      const b = canvas.getBoundingClientRect();
      pts.push([(e.clientX - b.left) * width / b.width, (e.clientY - b.top) * height / b.height]);
      redraw();
      if (pts.length === 3) { canvas.onclick = null; resolve(pts); }
    };
  });
}
"""


def pick_hoop(bgr_frame) -> dict:
    """Shows the frame in the notebook output and waits for 3 clicks. Returns a hoop.json dict."""
    import cv2
    from google.colab import output
    from IPython.display import Javascript, display

    h, w = bgr_frame.shape[:2]
    ok, jpg = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("could not encode frame")
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii")
    display(Javascript(_JS))
    left, right, front = output.eval_js(f"pickHoop('{data_url}', {w}, {h})", timeout_sec=None)
    return {
        "image_width": w,
        "image_height": h,
        "left": [round(left[0], 1), round(left[1], 1)],
        "right": [round(right[0], 1), round(right[1], 1)],
        "front": [round(front[0], 1), round(front[1], 1)],
    }
