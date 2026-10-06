/* The robot viewer of the geodex docs.
 *
 * Each .geodex-robot-scene element names a scene file (data-scene) and the directory of the
 * robot models (data-robots). A scene holds a path of joint configurations in the robot's
 * joint order, sampled at a fixed rate, the obstacles, the traced paths and the camera. A
 * robot model holds the kinematic tree (chain.json) and the visual meshes (GLB files). The
 * viewer plays the path in a loop with translucent copies of the robot along it, and the
 * reader can orbit the camera, pause and scrub. With data-spheres="1" the robot also shows
 * its VAMP collision spheres (spheres.json), and with data-spheres="hidden" the spheres start
 * hidden. The Spheres button shows and hides them.
 *
 * A scene without a robot holds the positions of small balls, the movers, three coordinates per
 * mover and frame. With "floor": null, the scene does not have a floor and the camera orbits
 * all the way around, as for a path on a sphere.
 *
 * A hero scene loads with the page and plays at once. Any other scene loads and plays when
 * it scrolls into view. Every scene pauses while it is off screen, and none plays by itself
 * when the reader prefers reduced motion. The poster image stays in place until the first
 * frame is drawn and remains the content when JavaScript or WebGL is off.
 *
 * z is up throughout, and lengths are in meters.
 */
import * as THREE from "../three/three.module.min.js";
import { GLTFLoader } from "../three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "../three/addons/libs/meshopt_decoder.module.js";
import { OrbitControls } from "../three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "../three/addons/environments/RoomEnvironment.js";
import { Line2 } from "../three/addons/lines/Line2.js";
import { LineMaterial } from "../three/addons/lines/LineMaterial.js";
import { LineGeometry } from "../three/addons/lines/LineGeometry.js";

const loader = new GLTFLoader();
loader.setMeshoptDecoder(MeshoptDecoder);
const cache = new Map();

const smooth = (x) => x * x * (3 - 2 * x);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function once(key, make) {
  if (!cache.has(key)) cache.set(key, make());
  return cache.get(key);
}

function loadJSON(url) {
  return once(url, () => fetch(url).then((r) => {
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }));
}

/* ------------------------------------------------------------------ robot model */

/* Meshes grouped by link, each with its transform relative to the link node. */
function linkMeshes(root) {
  const out = {};
  root.updateMatrixWorld(true);
  root.traverse((o) => {
    if (!o.isMesh) return;
    const link = o.name.replace(/__\d+$/, "");
    const node = root.getObjectByName(link);
    const rel = new THREE.Matrix4();
    if (node) rel.copy(node.matrixWorld).invert();
    rel.multiply(o.matrixWorld);
    (out[link] = out[link] || []).push({ geometry: o.geometry, material: o.material, matrix: rel });
  });
  return out;
}

function tuneMaterial(material) {
  const m = material.clone();
  m.envMapIntensity = 1.0;
  if (m.emissive) m.emissiveIntensity = 0.35;
  return m;
}

const _m = new THREE.Matrix4();
const _v = new THREE.Vector3();

export class RobotModel {
  constructor(chain, full, ghost, spheres) {
    this.chain = chain;
    this.full = full;
    this.ghost = ghost;
    this.spheres = spheres;
    this.dof = chain.joint_names.length;
    this.joints = chain.joints.map((j) => ({
      ...j,
      origin: new THREE.Matrix4().compose(new THREE.Vector3(...j.origin.xyz),
        new THREE.Quaternion(...j.origin.quat), new THREE.Vector3(1, 1, 1)),
      axisV: new THREE.Vector3(...j.axis).normalize(),
      multiplier: j.multiplier ?? 1,
      offset: j.offset ?? 0,
    }));
    const ee = chain.ee;
    this.eeOffset = new THREE.Matrix4().compose(new THREE.Vector3(...ee.xyz),
      new THREE.Quaternion(...ee.quat), new THREE.Vector3(1, 1, 1));
  }

  /* Local transform of joint j at configuration q. */
  local(j, q, out) {
    const value = q[j.index] * j.multiplier + j.offset;
    if (j.type === "prismatic") _m.makeTranslation(_v.copy(j.axisV).multiplyScalar(value));
    else _m.makeRotationAxis(j.axisV, value);
    return out.copy(j.origin).multiply(_m);
  }

  /* World matrix of every moving link at q, without a scene graph. */
  linkMatrices(q) {
    const out = { [this.chain.root]: new THREE.Matrix4() };
    const local = new THREE.Matrix4();
    for (const j of this.joints) {
      out[j.child] = out[j.parent].clone().multiply(this.local(j, q, local));
    }
    return out;
  }

  eeMatrix(q) {
    return this.linkMatrices(q)[this.chain.ee.parent].multiply(this.eeOffset);
  }

  /* Link origins and the end effector at q, for framing the camera. */
  keyPoints(q) {
    const links = this.linkMatrices(q);
    const pts = Object.values(links).map((m) => new THREE.Vector3().setFromMatrixPosition(m));
    pts.push(new THREE.Vector3().setFromMatrixPosition(this.eeMatrix(q)));
    return pts;
  }
}

export function loadModel(base, name, withSpheres) {
  return once(`${base}${name}|${withSpheres ? 1 : 0}`, async () => {
    const dir = `${base}${name}/`;
    const chain = await loadJSON(`${dir}chain.json`);
    const [full, ghost, spheres] = await Promise.all([
      loader.loadAsync(dir + chain.meshes.full),
      loader.loadAsync(dir + chain.meshes.ghost),
      withSpheres ? loadJSON(`${dir}spheres.json`) : Promise.resolve(null),
    ]);
    const meshes = linkMeshes(full.scene);
    const tuned = new Map();
    Object.values(meshes).flat().forEach((m) => {
      if (!tuned.has(m.material)) tuned.set(m.material, tuneMaterial(m.material));
      m.material = tuned.get(m.material);
    });
    return new RobotModel(chain, meshes, linkMeshes(ghost.scene), spheres);
  });
}

/* A posable robot. The full level of detail keeps the real materials, and the ghost level
 * shares one given material. */
export class Robot extends THREE.Group {
  constructor(model, { ghost = false, material = null, spheres = null } = {}) {
    super();
    this.model = model;
    this.links = {};
    this._local = new THREE.Matrix4();
    const meshes = ghost ? model.ghost : model.full;
    const root = new THREE.Group();
    this.add(root);
    this.links[model.chain.root] = root;
    this._attach(root, meshes[model.chain.root], material, !ghost);
    for (const j of model.joints) {
      const link = new THREE.Group();
      link.matrixAutoUpdate = false;
      this.links[j.parent].add(link);
      this.links[j.child] = link;
      this._attach(link, meshes[j.child], material, !ghost);
    }
    this.spheres = [];
    if (spheres && model.spheres) this._addSpheres(model.spheres, spheres);
  }

  _attach(link, meshes, material, castShadow) {
    (meshes || []).forEach((src) => {
      const mesh = new THREE.Mesh(src.geometry, material || src.material);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(src.matrix);
      mesh.castShadow = castShadow;
      link.add(mesh);
    });
  }

  _addSpheres(table, material) {
    const geometry = new THREE.SphereGeometry(1, 20, 14);
    const m = new THREE.Matrix4();
    for (const [link, balls] of Object.entries(table)) {
      if (!this.links[link]) continue;
      const mesh = new THREE.InstancedMesh(geometry, material, balls.length);
      balls.forEach(([x, y, z, r], i) => {
        mesh.setMatrixAt(i, m.makeScale(r, r, r).setPosition(x, y, z));
      });
      mesh.renderOrder = 5;
      this.links[link].add(mesh);
      this.spheres.push(mesh);
    }
  }

  setQ(q) {
    for (const j of this.model.joints) {
      const link = this.links[j.child];
      this.model.local(j, q, link.matrix);
      link.matrixWorldNeedsUpdate = true;
    }
    return this;
  }

  showSpheres(on) {
    this.spheres.forEach((s) => { s.visible = on; });
  }
}

function ghostMaterial(color, opacity) {
  const c = new THREE.Color(color);
  return new THREE.MeshStandardMaterial({
    color: c, emissive: c.clone().multiplyScalar(0.18), roughness: 0.75, metalness: 0.0,
    transparent: true, opacity, depthWrite: true, polygonOffset: true,
    polygonOffsetFactor: 2, polygonOffsetUnits: 6,
  });
}

/* ------------------------------------------------------------------ stage */

/* Grid on z = 0 with major lines every `major` cells, fading out towards its rim. */
function makeGrid(half, cell, major, center) {
  const pos = [], col = [];
  const minor = new THREE.Color("#dddde3"), strong = new THREE.Color("#c6c6ce");
  const bg = new THREE.Color("#ffffff");
  const n = Math.round(half / cell);
  const piece = cell / 2;
  const fade = (x, y) => 1 - smooth(clamp((Math.hypot(x, y) / half - 0.45) / 0.55, 0, 1));
  const push = (x0, y0, x1, y1, c) => {
    const f0 = fade(x0, y0), f1 = fade(x1, y1);
    if (f0 <= 0.001 && f1 <= 0.001) return;
    pos.push(x0 + center[0], y0 + center[1], 0, x1 + center[0], y1 + center[1], 0);
    const a = bg.clone().lerp(c, f0), b = bg.clone().lerp(c, f1);
    col.push(a.r, a.g, a.b, b.r, b.g, b.b);
  };
  for (let i = -n; i <= n; i++) {
    const c = i % major === 0 ? strong : minor;
    const s = i * cell;
    for (let t = -half; t < half - 1e-9; t += piece) {
      push(s, t, s, t + piece, c);
      push(t, s, t + piece, s, c);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true }));
  lines.position.z = 0.0008;
  lines.renderOrder = -1;
  return lines;
}

class Stage {
  constructor(host, { floor, label, interactive, fov }) {
    this.host = host;
    this.animators = new Set();
    this.lineMaterials = new Set();
    this.visible = true;
    this.frame = 0;
    this.last = 0;

    const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: !interactive });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.VSMShadowMap;
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.setAttribute("aria-label", label);
    renderer.domElement.classList.add("geodex-robot-canvas");
    host.appendChild(renderer.domElement);
    this.renderer = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#ffffff");
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    scene.environmentIntensity = 0.55;
    this.scene = scene;

    const open = floor === null;
    floor = floor || {};
    const center = floor.center || [0, 0];
    const half = floor.half || 1.5;
    const z0 = floor.z || 0;
    scene.add(new THREE.HemisphereLight(0xffffff, 0xe9e9ee, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    const height = Math.max(3.2, 2.2 * half);
    key.position.set(center[0] + 0.2 * height, center[1] - 0.13 * height, z0 + height);
    key.target.position.set(center[0], center[1], z0);
    key.castShadow = !open;
    key.shadow.mapSize.set(1024, 1024);
    const e = 1.1 * half;
    Object.assign(key.shadow.camera, { left: -e, right: e, top: e, bottom: -e, near: 0.2,
      far: 2 * height + 2 });
    key.shadow.radius = 6;
    key.shadow.blurSamples = 16;
    key.shadow.bias = -0.0004;
    scene.add(key, key.target);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(center[0] - 2, center[1] + 2, 1.2);
    scene.add(fill);

    if (!open) {
      const ground = new THREE.Mesh(new THREE.PlaneGeometry(6 * half, 6 * half),
        new THREE.ShadowMaterial({ opacity: 0.13, depthWrite: false }));
      ground.position.set(center[0], center[1], z0);
      ground.receiveShadow = true;
      ground.renderOrder = -2;
      scene.add(ground);
      const grid = makeGrid(half, floor.cell || 0.1, floor.major || 5, center);
      grid.position.z += z0;
      scene.add(grid);
    }

    const camera = new THREE.PerspectiveCamera(fov, 1, 0.02, 60);
    camera.up.set(0, 0, 1);
    this.camera = camera;

    if (interactive) {
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.12;
      controls.enablePan = false;
      controls.enableZoom = false;
      controls.maxPolarAngle = open ? Math.PI : Math.PI * 0.495;
      controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_ROTATE };
      controls.addEventListener("change", () => this.invalidate());
      controls.addEventListener("start", () => host.classList.add("geodex-robot-touched"));
      renderer.domElement.style.touchAction = "pan-y";
      renderer.domElement.addEventListener("pointerdown", () => { controls.enableZoom = true; });
      renderer.domElement.addEventListener("pointerleave", () => { controls.enableZoom = false; });
      this.controls = controls;
    }

    this._tick = this._tick.bind(this);
    this.resize();
    if ("ResizeObserver" in window) new ResizeObserver(() => this.resize()).observe(host);
    else window.addEventListener("resize", () => this.resize());
  }

  look(position, target) {
    this.camera.position.set(...position);
    this.camera.lookAt(new THREE.Vector3(...target));
    if (this.controls) {
      this.controls.target.set(...target);
      const d = this.camera.position.distanceTo(this.controls.target);
      this.controls.minDistance = 0.4 * d;
      this.controls.maxDistance = 2.2 * d;
      this.controls.update();
    }
    this.invalidate();
  }

  resize() {
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    if (w === this.w && h === this.h) return;
    this.w = w;
    this.h = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.lineMaterials.forEach((m) => m.resolution.set(w, h));
    this.invalidate();
  }

  lineMaterial(options) {
    const m = new LineMaterial({ linewidth: 2.5, worldUnits: false, transparent: true,
      depthWrite: false, ...options });
    m.resolution.set(this.w || 1, this.h || 1);
    this.lineMaterials.add(m);
    return m;
  }

  invalidate() {
    if (!this.frame) this.frame = requestAnimationFrame(this._tick);
  }

  /* fn(dt) runs every frame and returns true to keep the loop going. */
  animate(fn) {
    this.animators.add(fn);
    this.last = 0;
    this.invalidate();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  _tick(now) {
    this.frame = 0;
    const dt = this.last ? Math.min(0.1, (now - this.last) / 1000) : 1 / 60;
    let again = false;
    if (this.visible) {
      for (const fn of Array.from(this.animators)) {
        if (fn(dt)) again = true;
        else this.animators.delete(fn);
      }
    }
    if (this.controls && this.controls.update()) again = true;
    if (this.visible) this.render();
    this.last = again && this.visible ? now : 0;
    if (again && this.visible) this.invalidate();
  }

  setVisible(on) {
    this.visible = on;
    if (on) this.invalidate();
  }
}

/* A fat polyline drawn up to a given point. */
class Polyline {
  constructor(stage, points, { color, width = 2.5, opacity = 1 }) {
    this.material = stage.lineMaterial({ color: new THREE.Color(color), linewidth: width, opacity });
    this.opacity = opacity;
    const g = new LineGeometry();
    g.setPositions(points.flat());
    this.line = new Line2(g, this.material);
    this.line.computeLineDistances();
    this.line.renderOrder = 3;
    this.count = points.length;
  }

  reveal(k) {
    this.line.geometry.instanceCount = clamp(Math.round(k) - 1, 0, this.count - 1);
  }

  fade(k) {
    this.material.opacity = this.opacity * k;
  }
}

/* ------------------------------------------------------------------ scene player */

const TIMES = { start: 0.6, end: 1.4 };

class Viewer {
  constructor(root) {
    this.root = root;
    this.mode = root.dataset.mode || "page";
    this.playing = false;
    this.clock = 0;
    this.q = null;
  }

  async init() {
    const root = this.root;
    const data = await loadJSON(root.dataset.scene);
    const withSpheres = root.dataset.spheres === "1" || root.dataset.spheres === "hidden";
    let model = null;
    if (data.robot) {
      model = await loadModel(root.dataset.robots, data.robot, withSpheres);
      if (model.dof !== data.joint_names.length
          || model.chain.joint_names.some((n, i) => n !== data.joint_names[i])) {
        throw new Error(`${root.dataset.scene}: joint order differs from ${data.robot}`);
      }
    }
    this.data = data;
    this.model = model;
    this.frames = data.frames;
    this.q = new Float64Array(this.frames[0].length);
    const interactive = this.mode === "page";
    const stage = new Stage(root.querySelector(".geodex-robot-stage"), {
      floor: data.floor === null ? null : data.floor || {}, label: root.dataset.alt || "",
      interactive, fov: (data.camera && data.camera.fov) || 40,
    });
    this.stage = stage;

    for (const obj of data.objects || []) stage.scene.add(this.object(obj));
    this.ghostAt = [];
    this.ghosts = [];
    const n = this.frames.length;
    if (model) this.addRobot(data, withSpheres);
    this.movers = (data.movers || []).map((m) => {
      const ball = new THREE.Mesh(new THREE.SphereGeometry(m.radius || 0.04, 32, 20),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(m.color), roughness: 0.45 }));
      ball.renderOrder = 4;
      stage.scene.add(ball);
      return ball;
    });

    this.traces = (data.traces || []).map((t) => {
      const line = new Polyline(stage, t.points, t);
      stage.scene.add(line.line);
      return line;
    });
    for (const m of data.markers || []) {
      const ball = new THREE.Mesh(new THREE.SphereGeometry(m.radius || 0.015, 20, 14),
        new THREE.MeshStandardMaterial({ color: new THREE.Color(m.color), roughness: 0.5 }));
      ball.position.set(...m.position);
      stage.scene.add(ball);
    }

    const cam = data.camera;
    stage.look(cam.position, cam.target);
    this.move = (n - 1) / (data.fps || 30);
    this.cycle = data.timing === "cycle";
    this.back = this.cycle ? 0 : Math.min(1.6, Math.max(0.8, 0.25 * this.move));
    this.period = this.cycle ? this.move : TIMES.start + this.move + TIMES.end + this.back;
    if (this.cycle) this.seek(0);
    else this.seek(this.period - this.back - 1e-6);
    stage.render();
    root.classList.add("geodex-robot-ready");
    if (interactive) this.controls();
  }

  /* The robot, the objects it holds and its translucent copies along the path. */
  addRobot(data, withSpheres) {
    const { model, stage, root } = this;
    const sphereMaterial = withSpheres ? new THREE.MeshStandardMaterial({
      color: new THREE.Color(data.sphere_color || "#3e5e92"), roughness: 0.6, metalness: 0.0,
      transparent: true, opacity: 0.42, depthWrite: false,
    }) : null;
    this.robot = new Robot(model, { spheres: sphereMaterial });
    if (root.dataset.spheres === "hidden") this.robot.showSpheres(false);
    const held = data.held || [];
    this.hold(this.robot, held);
    stage.scene.add(this.robot);

    const ghosts = data.ghosts || { count: 0 };
    const n = this.frames.length;
    for (let i = 0; i < ghosts.count; i++) {
      const at = Math.round((i * (n - 1)) / ghosts.count);
      const t = ghosts.count > 1 ? i / (ghosts.count - 1) : 1;
      const material = ghostMaterial(ghosts.color || "#9fb0c8",
        (ghosts.from ?? 0.12) + ((ghosts.to ?? 0.35) - (ghosts.from ?? 0.12)) * t);
      const g = new Robot(model, { ghost: true, material }).setQ(this.frames[at]);
      this.hold(g, held, material);
      g.userData.base = material.opacity;
      g.renderOrder = 1 + i;
      stage.scene.add(g);
      this.ghosts.push(g);
      this.ghostAt.push(at);
    }
  }

  object(obj) {
    const color = new THREE.Color(obj.color || "#b8b2a8");
    const opacity = obj.opacity ?? 1;
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0.0,
      transparent: opacity < 1, opacity, depthWrite: opacity >= 1 });
    let geometry;
    if (obj.shape === "box") geometry = new THREE.BoxGeometry(...obj.size);
    else if (obj.shape === "cap") {
      /* The cap of half-angle `angle` around the unit direction `axis` on a sphere. */
      geometry = new THREE.SphereGeometry(obj.radius, 96, 48, 0, 2 * Math.PI, 0, obj.angle);
      geometry.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0), new THREE.Vector3(...obj.axis).normalize()));
    }
    else if (obj.shape === "cylinder") {
      geometry = new THREE.CylinderGeometry(obj.radius, obj.radius, obj.length, 40);
      geometry.rotateX(Math.PI / 2);
    } else geometry = new THREE.SphereGeometry(obj.radius, 32, 20);
    if (obj.shape === "cap") material.side = THREE.DoubleSide;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...(obj.position || [0, 0, 0]));
    if (obj.quat) mesh.quaternion.set(...obj.quat);
    mesh.castShadow = opacity >= 1;
    mesh.receiveShadow = true;
    return mesh;
  }

  /* Objects the robot holds, posed in its end-effector frame. A translucent copy draws them
   * in its own material. */
  hold(robot, objects, material = null) {
    if (!objects.length) return;
    const frame = new THREE.Group();
    frame.matrixAutoUpdate = false;
    frame.matrix.copy(this.model.eeOffset);
    for (const obj of objects) {
      const mesh = this.object(obj);
      if (material) {
        mesh.material = material;
        mesh.castShadow = false;
      }
      frame.add(mesh);
    }
    robot.links[this.model.chain.ee.parent].add(frame);
  }

  /* Pose everything at time t of the loop. The loop pauses at the start, follows the path,
   * pauses at the end, and returns along the path while the trail fades. */
  seek(t) {
    t = ((t % this.period) + this.period) % this.period;
    this.clock = t;
    if (this.cycle) {
      this.pose(t / this.period, 1);
      return;
    }
    const { start, end } = TIMES;
    let alpha, fade = 1;
    if (t < start) {
      alpha = 0;
      fade = 0;
    } else if (t < start + this.move) {
      alpha = smooth((t - start) / this.move);
    } else if (t < start + this.move + end) {
      alpha = 1;
    } else {
      const k = (t - start - this.move - end) / this.back;
      alpha = 1 - smooth(k);
      fade = 1 - smooth(Math.min(1, k * 1.4));
    }
    this.pose(alpha, fade);
  }

  pose(alpha, fade) {
    const n = this.frames.length;
    const f = alpha * (n - 1);
    const i = Math.min(n - 2, Math.floor(f)), w = f - i;
    const a = this.frames[Math.max(0, i)], b = this.frames[Math.min(n - 1, i + 1)];
    for (let k = 0; k < this.q.length; k++) this.q[k] = a[k] + (b[k] - a[k]) * w;
    if (this.robot) this.robot.setQ(this.q);
    this.movers.forEach((m, k) => m.position.set(this.q[3 * k], this.q[3 * k + 1],
      this.q[3 * k + 2]));
    const reached = fade >= 1 ? f : n - 1;
    this.traces.forEach((t) => { t.reveal(reached + 1); t.fade(fade); });
    this.ghosts.forEach((g, k) => {
      g.visible = fade > 0.01 && this.ghostAt[k] <= reached + 0.5;
      g.traverse((o) => {
        if (o.isMesh) o.material.opacity = g.userData.base * fade;
      });
    });
    if (this.scrub && !this.scrubbing) this.scrub.value = String(Math.round(alpha * 1000));
    this.stage.invalidate();
  }

  play() {
    if (this.playing) return;
    this.playing = true;
    this.root.classList.add("geodex-robot-playing");
    this.label("Pause");
    this.stage.animate((dt) => {
      if (!this.playing) return false;
      this.seek(this.clock + dt);
      return true;
    });
  }

  pause() {
    this.playing = false;
    this.root.classList.remove("geodex-robot-playing");
    this.label("Play");
  }

  label(text) {
    const button = this.root.querySelector(".geodex-robot-play");
    if (button) button.setAttribute("aria-label", text);
  }

  controls() {
    const bar = this.root.querySelector(".geodex-robot-controls");
    const button = bar.querySelector(".geodex-robot-play");
    button.addEventListener("click", () => {
      if (this.playing) this.pause();
      else this.play();
    });
    const scrub = bar.querySelector(".geodex-robot-scrub");
    this.scrub = scrub;
    scrub.addEventListener("input", () => {
      this.pause();
      this.scrubbing = true;
      const alpha = Number(scrub.value) / 1000;
      this.clock = this.cycle ? alpha * this.period
        : TIMES.start + this.move * inverseSmooth(alpha);
      this.pose(alpha, 1);
      this.scrubbing = false;
    });
    const toggle = bar.querySelector(".geodex-robot-spheres");
    if (toggle) {
      toggle.addEventListener("click", () => {
        const on = toggle.getAttribute("aria-pressed") !== "true";
        toggle.setAttribute("aria-pressed", on ? "true" : "false");
        this.robot.showSpheres(on);
        this.stage.invalidate();
      });
    }
    bar.hidden = false;
  }
}

/* The inverse of smooth on [0, 1], by bisection. */
function inverseSmooth(y) {
  let lo = 0, hi = 1;
  for (let i = 0; i < 30; i++) {
    const mid = 0.5 * (lo + hi);
    if (smooth(mid) < y) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/* ------------------------------------------------------------------ page wiring */

const viewers = [];
window.geodexRobotViewers = viewers;

function start(root) {
  if (root.dataset.started) return;
  root.dataset.started = "1";
  const viewer = new Viewer(root);
  viewers.push(viewer);
  viewer.ready = viewer.init().then(() => {
    if (viewer.mode !== "page") return viewer;
    const autoplay = !reducedMotion();
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => {
        const on = entries.some((e) => e.isIntersecting);
        viewer.stage.setVisible(on);
        if (on && autoplay && !viewer.stopped) viewer.play();
      }, { threshold: 0.15 }).observe(root);
    } else if (autoplay) viewer.play();
    root.querySelector(".geodex-robot-play").addEventListener("click", () => {
      viewer.stopped = !viewer.playing;
    });
    root.querySelector(".geodex-robot-scrub").addEventListener("input", () => {
      viewer.stopped = true;
    });
    return viewer;
  }).catch((err) => {
    root.classList.add("geodex-robot-failed");
    console.warn("robot viewer:", err);
    return viewer;
  });
}

function webgl() {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch (e) {
    return false;
  }
}

function wire() {
  const scenes = document.querySelectorAll(".geodex-robot-scene");
  if (!scenes.length || !webgl()) return;
  const near = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        near.unobserve(e.target);
        start(e.target);
      }
    });
  }, { rootMargin: "300px 0px" }) : null;
  scenes.forEach((root) => {
    if (root.dataset.hero === "1" || root.dataset.mode !== undefined || !near) start(root);
    else near.observe(root);
  });
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire);
else wire();
