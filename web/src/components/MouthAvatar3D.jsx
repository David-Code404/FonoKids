import { useEffect, useRef } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/examples/jsm/libs/meshopt_decoder.module.js";
import "./MouthAvatar3D.css";

// Los landmarks se grabaron a 25fps (ver grabar_video_continuo.py), pero
// renderizamos al doble de esa cadencia e interpolamos linealmente entre
// cada par de frames reales consecutivos -- así el movimiento se ve fluido
// en vez de "a los saltos", sin inventar ningún dato (son los mismos puntos
// medidos, solo que dibujados en posiciones intermedias reales).
const DATA_FPS = 25;
const RENDER_FPS = 50;
const RENDER_MS = 1000 / RENDER_FPS;

// african.glb: personaje "African girl Rigged" de TurboSquid (gratis),
// exportado desde Blender (tools/export_african_optimized.py) -- rig
// completo estilo Reallusion Character Creator, con deformación 3D REAL de
// boca (Jaw_Open, Mouth_Funnel_*, Mouth_Pucker_*, Mouth_Stretch_*, visemas
// V_*), no basada en textura como vroid.glb (que probamos y no servía).
// Se le quitó la ropa (pantalón/camisa) y se bajaron las texturas a 1024px
// -- de 601MB originales a ~50MB. OJO: revisar la licencia exacta de
// TurboSquid antes de publicar la app (este modelo es "Free" pero cada uno
// tiene sus propios términos).
const MODEL_URL = "/models/african.glb";
const APPLY_GENERIC_TOON_SHADING = false;

// Índices dentro del array de 20 puntos de labios (iBUG 48-67, ver
// normalize_lip_landmarks() en server/main.py): contorno externo = 0..11
// (48..59), contorno interno = 12..19 (60..67).
const OUTER_L = 0;
const OUTER_R = 6;
const INNER_TOP = 14;
const INNER_BOTTOM = 18;

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

// Textura de 4 tonos para MeshToonMaterial -- convierte el sombreado suave
// tipo PBR en bandas planas de luz/sombra (look "muñeco"/cel-shading), sin
// tocar las texturas originales (piel, cejas, ojos siguen siendo las mismas,
// solo cambia CÓMO se iluminan).
function makeGradientMap() {
  const data = new Uint8Array([70, 140, 200, 255]);
  const texture = new THREE.DataTexture(data, data.length, 1, THREE.RedFormat);
  texture.needsUpdate = true;
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  return texture;
}

// Reemplaza el material PBR (realista) de cada malla por un MeshToonMaterial
// -- conserva mapas de textura/alpha (así cejas, pestañas y ropa, que usan
// recorte por transparencia, no se rompen) y flags de morph/skinning, pero
// cambia el sombreado a bandas planas para un look más caricatura/animado.
function toonifyMaterial(mat, gradientMap) {
  return new THREE.MeshToonMaterial({
    color: mat.color ? mat.color.clone() : new THREE.Color(0xffffff),
    map: mat.map || null,
    alphaMap: mat.alphaMap || null,
    transparent: mat.transparent,
    alphaTest: mat.alphaTest,
    side: mat.side,
    skinning: !!mat.skinning,
    morphTargets: !!mat.morphTargets,
    morphNormals: !!mat.morphNormals,
    gradientMap,
  });
}

// Interpola linealmente entre los 20 puntos del frame `a` y del frame `b`.
function lerpPoints(a, b, t) {
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    out[i] = [a[i][0] + (b[i][0] - a[i][0]) * t, a[i][1] + (b[i][1] - a[i][1]) * t];
  }
  return out;
}

// Posición de reproducción en vaivén (ida y vuelta) sobre [0, n-1]: evita el
// salto brusco que se ve al cortar del último frame de vuelta al primero
// cuando el clip de referencia hace loop.
function pingPongIndex(t, n) {
  if (n <= 1) return 0;
  const period = (n - 1) * 2;
  const m = t % period;
  return m <= n - 1 ? m : period - m;
}

// Constantes calibradas con datos reales de un clip "perro_correcto" (108
// frames, ver GET /reference/perro): innerH (apertura interna de labios) va
// de ~0.03 (boca cerrada) a ~0.25 (bien abierta); outerW (ancho externo)
// promedia ~0.57 con desvío de ~0.06-0.08 hacia cada lado. No son datos
// inventados, son el rango que efectivamente mide el pipeline de landmarks.
function computeBlendWeights(pts) {
  const innerH = dist(pts[INNER_TOP], pts[INNER_BOTTOM]);
  const outerW = dist(pts[OUTER_L], pts[OUTER_R]);

  const jawOpen = clamp((innerH - 0.025) / 0.15, 0, 1);
  const stretch = clamp((outerW - 0.57) / 0.07, 0, 1);
  const pucker = clamp((0.57 - outerW) / 0.05, 0, 1);

  return {
    jawOpen,
    mouthOpen: jawOpen,
    // Sumamos el despegue de labio inferior/superior al mismo jawOpen --
    // en este rig el hueso de la mandíbula solo mueve un poco la malla de
    // labios, así que sin esto la boca se ve casi cerrada aunque jawOpen
    // esté al máximo.
    mouthLowerDownLeft: jawOpen * 0.8,
    mouthLowerDownRight: jawOpen * 0.8,
    mouthUpperUpLeft: jawOpen * 0.4,
    mouthUpperUpRight: jawOpen * 0.4,
    mouthStretchLeft: stretch,
    mouthStretchRight: stretch,
    mouthFunnel: pucker,
    mouthPucker: pucker * 0.7,
    // Formas de vocales nativas de VRoid (Fcl_MTH_*) -- si el modelo
    // cargado las tiene (vroid.glb), se usan además de jawOpen; si no
    // existen en el diccionario de morph targets simplemente se ignoran
    // (ver applyCurrentFrame, que solo aplica nombres que encuentra).
    Fcl_MTH_A: jawOpen,
    Fcl_MTH_O: pucker * jawOpen,
    Fcl_MTH_U: pucker,
    Fcl_MTH_I: stretch,
    // Nombres estilo Reallusion Character Creator (african.glb) -- mismo
    // criterio: solo se aplican si existen en el diccionario del modelo.
    Jaw_Open: jawOpen,
    Mouth_Down_Lower_L: jawOpen * 0.8,
    Mouth_Down_Lower_R: jawOpen * 0.8,
    Mouth_Up_Upper_L: jawOpen * 0.4,
    Mouth_Up_Upper_R: jawOpen * 0.4,
    Mouth_Stretch_L: stretch,
    Mouth_Stretch_R: stretch,
    Mouth_Funnel_Up_L: pucker,
    Mouth_Funnel_Up_R: pucker,
    Mouth_Funnel_Down_L: pucker,
    Mouth_Funnel_Down_R: pucker,
    Mouth_Pucker_Up_L: pucker * 0.7,
    Mouth_Pucker_Up_R: pucker * 0.7,
    Mouth_Pucker_Down_L: pucker * 0.7,
    Mouth_Pucker_Down_R: pucker * 0.7,
    V_Open: jawOpen,
    V_Wide: stretch,
    V_Tight_O: pucker,
  };
}

// Formas de LENGUA del modelo (malla CC_Base_Tongue) -- a diferencia de las
// de labios/mandíbula de arriba, esto NO sale de ningún dato medido: la
// cámara nunca ve la lengua, es físicamente imposible. Es una animación
// ILUSTRATIVA (a mano, según fonética real) de dónde va la lengua en el
// momento de la "r" -- mismo criterio y mismo propósito que
// TonguePositionDiagram.jsx, acá integrado en la cabeza 3D. Nunca se
// presenta como "esto hizo tu lengua", siempre como ejemplo de cómo se dice.
const TONGUE_TAP_SHAPES = {
  Tongue_Tip_Up: 1,
  Tongue_Up: 0.5,
  Jaw_Open: 0.35,
};
// Dónde cae el "toque" de lengua dentro del ciclo de reproducción (0..1) --
// no hay forma de saber el instante exacto de la "r" sin datos reales, así
// que se ubica cerca de la mitad del clip como aproximación razonable para
// una animación de ejemplo, no una medición precisa.
const TONGUE_TAP_CENTER = 0.42;
const TONGUE_TAP_WIDTH = 0.1;

function tongueTapWeight(u) {
  const d = (u - TONGUE_TAP_CENTER) / TONGUE_TAP_WIDTH;
  return Math.exp(-(d * d));
}

/// Cabeza 3D (avatar Ready Player Me) cuya boca se anima cuadro a cuadro con
/// los landmarks REALES de labios de un clip "<palabra>_correcto" -- ver
/// GET /reference/<palabra> en server/main.py. El resto de la cara queda
/// estático; solo la boca se mueve, mapeando geometría real (apertura,
/// ancho) a los blendshapes ARKit del modelo (jawOpen, mouthStretch,
/// mouthFunnel/Pucker). Además, un toque de lengua ILUSTRATIVO (no medido,
/// ver TONGUE_TAP_SHAPES) para mostrar dónde va la lengua en la "r".
export default function MouthAvatar3D({ frames, size = 260 }) {
  const containerRef = useRef(null);
  const framesRef = useRef(frames);
  const mouthMeshesRef = useRef([]);

  framesRef.current = frames;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    let disposed = false;
    let timerId = null;
    let playbackT = 0;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(28, 1, 0.01, 10);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(size, size);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    // three.js >=0.155 usa iluminación físicamente correcta por defecto --
    // intensidades "1" quedan muy tenues/lavadas, hay que subirlas bastante
    // y fijar el tone mapping para que los colores de piel no salgan planos.
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.innerHTML = "";
    container.appendChild(renderer.domElement);

    // Con MeshToonMaterial una luz muy fuerte empuja todo a la banda más
    // clara del degradé y se pierde el efecto de bandas planas -- por eso
    // acá las intensidades son bastante más bajas que las que se usaban con
    // el material PBR realista.
    scene.add(new THREE.HemisphereLight(0xfff3e0, 0x554433, 1.3));
    const key = new THREE.DirectionalLight(0xffffff, 1.7);
    key.position.set(0.6, 0.8, 1);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffe0c2, 0.7);
    fill.position.set(-0.8, 0.2, 0.6);
    scene.add(fill);

    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);
    loader.load(
      MODEL_URL,
      (gltf) => {
        if (disposed) return;
        const root = gltf.scene;
        scene.add(root);

        mouthMeshesRef.current = [];
        const gradientMap = makeGradientMap();
        root.traverse((obj) => {
          // mpfb.glb reparte los blendshapes de boca en varias mallas (cara,
          // dientes, lengua, cejas, pestañas) -- hay que mover TODAS las que
          // tengan alguno de nuestros nombres de blendshape, si no algunas
          // quedan fijas mientras el resto de la boca se mueve.
          if (obj.isMesh && obj.morphTargetDictionary) {
            mouthMeshesRef.current.push(obj);
          }
          if (APPLY_GENERIC_TOON_SHADING && obj.isMesh && obj.material) {
            obj.material = Array.isArray(obj.material)
              ? obj.material.map((m) => toonifyMaterial(m, gradientMap))
              : toonifyMaterial(obj.material, gradientMap);
          }
        });

        // Encuadre: mpfb.glb es un cuerpo ENTERO (con ropa y pelo) donde la
        // malla "Human" incluye TODO el cuerpo (los blendshapes de cara
        // viven en la misma malla que el resto) -- su bounding box es la
        // del cuerpo completo, no sirve para encuadrar solo la cara. En vez
        // de eso usamos el hueso "Jaw" (mandíbula) del esqueleto, que está
        // ubicado justo en la boca, como punto de referencia real.
        const jawBone =
          root.getObjectByName("CC_Base_JawRoot") || root.getObjectByName("Jaw");
        const headBone =
          root.getObjectByName("CC_Base_Head") || root.getObjectByName("Head");
        const target = new THREE.Vector3();
        if (jawBone) {
          jawBone.getWorldPosition(target);
        } else if (headBone) {
          // Sin hueso de mandíbula (ej. vroid.glb): el origen de "Head"
          // suele estar en la base del cráneo/cuello, no a la altura de la
          // boca -- bajamos un poco el punto de mira para apuntar a la cara.
          headBone.getWorldPosition(target);
          target.y += 0.03;
        } else {
          new THREE.Box3().setFromObject(root).getCenter(target);
        }
        const HEAD_DISTANCE = 0.55;
        camera.position.set(target.x, target.y + 0.02, target.z + HEAD_DISTANCE);
        camera.lookAt(target);

        renderLoop();
      },
      undefined,
      (err) => console.error("No se pudo cargar el avatar 3D:", err)
    );

    function applyCurrentFrame() {
      const meshes = mouthMeshesRef.current;
      const currentFrames = framesRef.current;
      if (meshes.length === 0 || !currentFrames || currentFrames.length === 0) return;

      const idx = pingPongIndex(playbackT, currentFrames.length);
      const i0 = Math.floor(idx);
      const i1 = Math.min(i0 + 1, currentFrames.length - 1);
      const t = idx - i0;
      const pts = t === 0 ? currentFrames[i0] : lerpPoints(currentFrames[i0], currentFrames[i1], t);

      const weights = computeBlendWeights(pts);

      // Capa ilustrativa de lengua (ver TONGUE_TAP_SHAPES) -- se SUMA sobre
      // los pesos reales de labios, no los reemplaza.
      const tapAmount = tongueTapWeight(idx / Math.max(currentFrames.length - 1, 1));
      for (const [name, base] of Object.entries(TONGUE_TAP_SHAPES)) {
        weights[name] = Math.max(weights[name] || 0, base * tapAmount);
      }

      for (const mesh of meshes) {
        const dict = mesh.morphTargetDictionary;
        const influences = mesh.morphTargetInfluences;
        for (const [name, value] of Object.entries(weights)) {
          const i = dict[name];
          if (i !== undefined) influences[i] = value;
        }
      }
    }

    // setTimeout en vez de requestAnimationFrame: rAF no dispara en algunos
    // entornos de automatización/headless (confirmado en el Browser pane),
    // y acá solo necesitamos una cadencia fija de FPS, no sincronía con el
    // refresh de pantalla.
    //
    // El avance usa tiempo REAL transcurrido (performance.now()), no un
    // paso fijo por tick -- si el hilo principal se traba un instante (ej.
    // la GPU ocupada prediciendo un intento), el próximo tick avanza lo que
    // realmente pasó en vez de quedar "atrasado" y after acumular un salto
    // brusco cuando vuelve a tener CPU. Resultado: la velocidad percibida
    // se mantiene estable incluso con hiccups, en vez de tironear.
    let lastTickMs = null;
    function renderLoop() {
      if (disposed) return;
      const now = performance.now();
      const deltaMs = lastTickMs === null ? RENDER_MS : Math.min(now - lastTickMs, RENDER_MS * 4);
      lastTickMs = now;
      applyCurrentFrame();
      playbackT += (deltaMs / 1000) * DATA_FPS;
      renderer.render(scene, camera);
      timerId = setTimeout(renderLoop, RENDER_MS);
    }

    return () => {
      disposed = true;
      if (timerId) clearTimeout(timerId);
      renderer.dispose();
      mouthMeshesRef.current = [];
      container.innerHTML = "";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [size]);

  if (!frames || frames.length === 0) return null;

  return <div ref={containerRef} className="mouth-avatar-3d" style={{ width: size, height: size }} />;
}
