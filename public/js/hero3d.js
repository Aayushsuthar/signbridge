// Hero 3D: a hand built from the same 21-point skeleton MediaPipe tracks, drawn as glowing particles
// that flow along the bones. It morphs between signs (open palm → 🤟 "I love you" → fist) and follows the mouse.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.186.1/build/three.module.js";

// MediaPipe hand topology
const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

// finger chains: [mcp, pip, dip, tip] indices and their straight + curled shapes
const OPEN = [
  [0, 0, 0],
  [-0.38, 0.28, 0.05], [-0.66, 0.55, 0.08], [-0.86, 0.8, 0.08], [-1.0, 1.02, 0.06],
  [-0.32, 1.0, 0], [-0.37, 1.42, 0], [-0.4, 1.68, 0], [-0.42, 1.9, 0],
  [0, 1.05, 0], [0, 1.52, 0], [0, 1.8, 0], [0, 2.04, 0],
  [0.3, 1.0, 0], [0.34, 1.42, 0], [0.37, 1.67, 0], [0.39, 1.88, 0],
  [0.56, 0.88, 0], [0.66, 1.2, 0], [0.72, 1.4, 0], [0.76, 1.58, 0],
];

function curl(pose, mcp, amount = 1) {
  const p = pose.map((v) => v.slice());
  const [x, y] = p[mcp];
  const curled = [
    [x, y + 0.28, 0.34],
    [x * 0.95, y + 0.06, 0.5],
    [x * 0.9, y - 0.14, 0.38],
  ];
  for (let k = 0; k < 3; k++) {
    for (let a = 0; a < 3; a++) p[mcp + 1 + k][a] += (curled[k][a] - p[mcp + 1 + k][a]) * amount;
  }
  return p;
}

const ILY = curl(curl(OPEN, 9), 13); // 🤟: thumb, index and pinky up
const FIST = (() => {
  let p = curl(curl(curl(curl(OPEN, 5), 9), 13), 17);
  p = p.map((v) => v.slice());
  p[2] = [-0.5, 0.55, 0.3];
  p[3] = [-0.28, 0.75, 0.52];
  p[4] = [-0.05, 0.8, 0.58]; // thumb across the fingers
  return p;
})();
const POSES = [OPEN, ILY, OPEN, FIST];

const PER_BONE = 64;
const DUST = 700;

const smooth = (x) => x * x * (3 - 2 * x);

export function startHero3D(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  container.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 0.2, 7.2);

  const group = new THREE.Group();
  group.position.y = -1.3;
  group.scale.setScalar(1.3);
  scene.add(group);

  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uPixel: { value: renderer.getPixelRatio() } },
    vertexShader: `
      attribute float size;
      attribute vec3 color;
      varying vec3 vColor;
      uniform float uPixel;
      void main() {
        vColor = color;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * uPixel * (7.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        float a = smoothstep(0.5, 0.0, d);
        gl_FragColor = vec4(vColor * a, a);
      }`,
  });

  // hand particles: flowing along each bone, plus a bright point on each joint
  const handCount = BONES.length * PER_BONE + 21;
  const handPos = new Float32Array(handCount * 3);
  const handCol = new Float32Array(handCount * 3);
  const handSize = new Float32Array(handCount);
  const seeds = new Float32Array(handCount).map(() => Math.random());
  const cyan = new THREE.Color("#5cf2ff");
  const violet = new THREE.Color("#8b5cf6");
  const pink = new THREE.Color("#ff4fd8");
  const tmp = new THREE.Color();
  for (let i = 0; i < handCount; i++) {
    const joint = i >= BONES.length * PER_BONE;
    handSize[i] = joint ? 38 : 8 + seeds[i] * 11;
  }
  const handGeo = new THREE.BufferGeometry();
  handGeo.setAttribute("position", new THREE.BufferAttribute(handPos, 3));
  handGeo.setAttribute("color", new THREE.BufferAttribute(handCol, 3));
  handGeo.setAttribute("size", new THREE.BufferAttribute(handSize, 1));
  group.add(new THREE.Points(handGeo, material));

  // orbit ring + dust
  const dustPos = new Float32Array(DUST * 3);
  const dustCol = new Float32Array(DUST * 3);
  const dustSize = new Float32Array(DUST);
  for (let i = 0; i < DUST; i++) {
    const ring = i < 260;
    const a = Math.random() * Math.PI * 2;
    const rad = ring ? 1.75 + (Math.random() - 0.5) * 0.12 : 2.2 + Math.random() * 3.5;
    const y = ring ? (Math.random() - 0.5) * 0.06 : (Math.random() - 0.5) * 5;
    dustPos.set([Math.cos(a) * rad, y + (ring ? 1 : 1), Math.sin(a) * rad], i * 3);
    tmp.copy(ring ? cyan : violet).lerp(pink, Math.random() * 0.6);
    dustCol.set([tmp.r * (ring ? 0.8 : 0.35), tmp.g * (ring ? 0.8 : 0.35), tmp.b * (ring ? 0.8 : 0.35)], i * 3);
    dustSize[i] = ring ? 4 + Math.random() * 4 : 2 + Math.random() * 5;
  }
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dustPos, 3));
  dustGeo.setAttribute("color", new THREE.BufferAttribute(dustCol, 3));
  dustGeo.setAttribute("size", new THREE.BufferAttribute(dustSize, 1));
  const dust = new THREE.Points(dustGeo, material);
  dust.rotation.x = 0.35;
  group.add(dust);

  const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
  const onMove = (e) => {
    mouse.tx = (e.clientX / innerWidth - 0.5) * 2;
    mouse.ty = (e.clientY / innerHeight - 0.5) * 2;
  };
  addEventListener("pointermove", onMove);

  function resize() {
    const { width, height } = container.getBoundingClientRect();
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  const joints = Array.from({ length: 21 }, () => new THREE.Vector3());
  const HOLD = 2.2;
  const MORPH = 1.1;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let running = true;
  let raf;
  const t0 = performance.now();

  function frame(now) {
    if (!running) return;
    const t = reduced ? 1.5 : (now - t0) / 1000;

    // pose morph
    const cycle = HOLD + MORPH;
    const idx = Math.floor(t / cycle) % POSES.length;
    const local = t % cycle;
    const k = local < HOLD ? 0 : smooth((local - HOLD) / MORPH);
    const from = POSES[idx];
    const to = POSES[(idx + 1) % POSES.length];
    for (let j = 0; j < 21; j++) {
      joints[j].set(
        from[j][0] + (to[j][0] - from[j][0]) * k,
        from[j][1] + (to[j][1] - from[j][1]) * k,
        from[j][2] + (to[j][2] - from[j][2]) * k,
      );
    }

    // particles stream along bones from wrist to fingertip
    let p = 0;
    for (const [a, b] of BONES) {
      const A = joints[a];
      const B = joints[b];
      for (let n = 0; n < PER_BONE; n++, p++) {
        const s = (seeds[p] + t * (0.18 + seeds[p] * 0.25)) % 1;
        const jitter = 0.05 * Math.sin(t * 3 + seeds[p] * 40);
        handPos[p * 3] = A.x + (B.x - A.x) * s + jitter;
        handPos[p * 3 + 1] = A.y + (B.y - A.y) * s + jitter * 0.6;
        handPos[p * 3 + 2] = A.z + (B.z - A.z) * s - jitter;
        const hue = (A.y + (B.y - A.y) * s) / 2.1;
        tmp.copy(violet).lerp(cyan, Math.min(1, Math.max(0, hue * 1.3))).lerp(pink, Math.max(0, hue - 0.7) * 1.6);
        const fade = Math.sin(s * Math.PI);
        handCol[p * 3] = tmp.r * fade;
        handCol[p * 3 + 1] = tmp.g * fade;
        handCol[p * 3 + 2] = tmp.b * fade;
      }
    }
    for (let j = 0; j < 21; j++, p++) {
      handPos.set([joints[j].x, joints[j].y, joints[j].z], p * 3);
      const pulse = 0.75 + 0.25 * Math.sin(t * 4 + j);
      tmp.copy(cyan).lerp(pink, j / 20);
      handCol.set([tmp.r * pulse, tmp.g * pulse, tmp.b * pulse], p * 3);
    }
    handGeo.attributes.position.needsUpdate = true;
    handGeo.attributes.color.needsUpdate = true;

    mouse.x += (mouse.tx - mouse.x) * 0.05;
    mouse.y += (mouse.ty - mouse.y) * 0.05;
    group.rotation.y = Math.sin(t * 0.45) * 0.45 + mouse.x * 0.5;
    group.rotation.x = -0.08 + mouse.y * 0.25;
    group.position.y = -1.3 + Math.sin(t * 0.9) * 0.06;
    dust.rotation.y = t * 0.12;

    renderer.render(scene, camera);
    if (!reduced) raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);

  return {
    stop() {
      running = false;
      cancelAnimationFrame(raf);
      ro.disconnect();
      removeEventListener("pointermove", onMove);
      renderer.dispose();
      handGeo.dispose();
      dustGeo.dispose();
      material.dispose();
      renderer.domElement.remove();
    },
  };
}
