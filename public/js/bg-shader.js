// Full-screen "liquid metal" background: domain-warped noise with chrome-like reflection bands,
// tinted with the app's aurora palette. Rendered at reduced resolution; pauses when hidden.

const VERT = `attribute vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;

const FRAG = `
precision highp float;
uniform vec2 r;
uniform float t;
uniform vec2 m;
uniform float glow;

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p){
  float v = 0.0, a = 0.5;
  mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
  for (int i = 0; i < 5; i++) { v += a * noise(p); p = rot * p * 2.0 + 3.0; a *= 0.5; }
  return v;
}

void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5 * r) / r.y;
  uv += (m - 0.5) * 0.12;
  float tt = t * 0.05;

  vec2 q = vec2(fbm(uv * 1.3 + tt), fbm(uv * 1.3 + vec2(5.2, 1.3) - tt));
  vec2 w = vec2(fbm(uv * 1.5 + 3.0 * q + vec2(1.7, 9.2) + tt * 1.3), fbm(uv * 1.5 + 3.0 * q + vec2(8.3, 2.8) - tt));
  float f = fbm(uv * 1.1 + 3.5 * w);

  // chrome: sharp repeating reflection bands that follow the flow
  float band = 0.5 + 0.5 * cos(6.2831 * (f * 2.4 + w.x * 0.7 + tt * 0.8));
  vec3 cyan = vec3(0.36, 0.95, 1.0);
  vec3 violet = vec3(0.55, 0.36, 0.98);
  vec3 pink = vec3(1.0, 0.31, 0.85);
  vec3 tint = mix(violet, cyan, smoothstep(0.25, 0.85, w.x));
  tint = mix(tint, pink, smoothstep(0.5, 0.95, w.y) * 0.75);

  vec3 base = vec3(0.018, 0.02, 0.04);
  float metal = pow(band, 3.0);
  vec3 col = base + tint * (metal * 0.55 + 0.08) * smoothstep(0.15, 0.95, f) * glow;
  col += vec3(1.0) * pow(band, 22.0) * 0.18 * glow;           // specular glints
  col *= 1.0 - 0.6 * dot(uv * 0.75, uv * 0.75);               // vignette
  gl_FragColor = vec4(col, 1.0);
}`;

export function startBackground(canvas) {
  const gl = canvas.getContext("webgl", { antialias: false, premultipliedAlpha: false });
  if (!gl) {
    canvas.remove();
    return { setGlow() {} };
  }
  const compile = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  gl.useProgram(prog);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const u = { r: gl.getUniformLocation(prog, "r"), t: gl.getUniformLocation(prog, "t"), m: gl.getUniformLocation(prog, "m"), glow: gl.getUniformLocation(prog, "glow") };

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mouse = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5 };
  let glow = 1;
  let glowTarget = 1;
  const scale = 0.5; // render at half resolution: it's soft anyway, and it keeps the GPU free for MediaPipe

  function resize() {
    const w = Math.max(1, Math.floor(innerWidth * scale));
    const h = Math.max(1, Math.floor(innerHeight * scale));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
    }
  }
  addEventListener("resize", resize);
  addEventListener("pointermove", (e) => {
    mouse.tx = e.clientX / innerWidth;
    mouse.ty = 1 - e.clientY / innerHeight;
  });
  resize();

  const t0 = performance.now();
  function frame(now) {
    mouse.x += (mouse.tx - mouse.x) * 0.04;
    mouse.y += (mouse.ty - mouse.y) * 0.04;
    glow += (glowTarget - glow) * 0.05;
    gl.uniform2f(u.r, canvas.width, canvas.height);
    gl.uniform1f(u.t, reduced ? 20 : (now - t0) / 1000);
    gl.uniform2f(u.m, mouse.x, mouse.y);
    gl.uniform1f(u.glow, glow);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (!reduced || Math.abs(glowTarget - glow) > 0.01) requestAnimationFrame(frame);
    else looping = false;
  }
  let looping = false;
  const kick = () => {
    if (looping) return;
    looping = true;
    requestAnimationFrame(frame); // rAF pauses by itself while the tab is hidden
  };
  kick();

  return {
    setGlow(v) {
      glowTarget = v;
      kick();
    },
  };
}
