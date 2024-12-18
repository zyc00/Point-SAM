import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { WebIO } from "https://cdn.skypack.dev/@gltf-transform/core";
import { KHRONOS_EXTENSIONS } from "https://cdn.skypack.dev/@gltf-transform/extensions";
import { metalRough } from "https://cdn.skypack.dev/@gltf-transform/functions";
import { sample } from "/static/sample_pc.js";

export var scene,
  canvas,
  canvas_width,
  canvas_height,
  camera,
  renderer,
  points,
  geometry,
  controls,
  origin_colors;

// Main function
async function main() {
  initialize_viewer();

  // ============================== Load point cloud ===================================
  // Change different point cloud here, all demo point clouds are saved in static/models
  // ===================================================================================

  await loadPointCloud("/pointcloud/089_0.ply");

  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    controls.update();
  }
  animate();
}

function createLights() {
  const ambientLight = new THREE.AmbientLight("white", 2);

  const mainLight = new THREE.DirectionalLight("white", 5);
  mainLight.position.set(10, 10, 10);

  return { ambientLight, mainLight };
}

function initialize_viewer() {
  scene = new THREE.Scene();

  // Add camera
  canvas = document.getElementById("viewer");
  canvas_width = canvas.getBoundingClientRect().width;
  canvas_height = canvas.getBoundingClientRect().height;
  camera = new THREE.OrthographicCamera(
    canvas_width / -800, // Left
    canvas_width / 800, // Right
    canvas_height / 800, // Top
    canvas_height / -800, // Bottom
    1, // Near
    1000 // Far
  );
  camera.position.z = 5;

  // Add renderer
  renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
  });
  renderer.setSize(canvas_width, canvas_height);

  // Add controls
  controls = new TrackballControls(camera, renderer.domElement);
  // controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.25;
  controls.enableZoom = true;
}

// Point cloud loader
async function loadPointCloud(ply_path) {
  var response = await fetch(ply_path);
  var data = await response.json();
  var positions = data.xyz;
  var colors = data.rgb;
  var material = new THREE.PointsMaterial({
    size: 10,
    vertexColors: true,
  });

  geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  points = new THREE.Points(geometry, material);
  origin_colors = points.geometry.attributes.color.clone();
  scene.add(points);
}

// Mesh Loader
async function loadMesh(mesh_path) {
  const io = new WebIO().registerExtensions(KHRONOS_EXTENSIONS);
  const document = await io.read(mesh_path);
  await document.transform(metalRough());
  const glb = await io.writeBinary(document);
  const { ambientLight, mainLight } = createLights();
  scene.add(ambientLight, mainLight);

  var loader = new GLTFLoader();
  loader.parse(glb.buffer, "", (gltf) => {
    const box = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());

    const scaleFactor = 1 / Math.max(size.x, size.y, size.z);
    const translation = center.multiplyScalar(-1);

    gltf.scene.traverse(function (object) {
      if (object.isMesh) {
        object.geometry.translate(translation.x, translation.y, translation.z);
        object.geometry.scale(scaleFactor, scaleFactor, scaleFactor);
      }
    });

    var sampled_points = sample(gltf.scene.children[0], 10000);
    var material = new THREE.PointsMaterial({
      size: 10,
      vertexColors: true,
    });

    points = new THREE.Points(sampled_points.geometry, material);
    origin_colors = points.geometry.attributes.color.clone();
    fetch("/sampled_pointcloud", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        points: sampled_points.geometry.attributes.position.array,
        colors: sampled_points.geometry.attributes.color.array,
      }),
    });
    scene.add(points);
    // scene.add(gltf.scene.children[0]);
  });
}
main();
