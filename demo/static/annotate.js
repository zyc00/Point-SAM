import * as THREE from "three";
import { camera, points, renderer, origin_colors } from "/static/viewer.js";

var positive_flag = true;
var drag_flag = false;
var raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.03;
var alpha = 0.4;
var mask_color = [0, 0, 1];
var prompts = [];
var labels = [];

function onPositiveClick() {
  positive_flag = true;
}

function onNegativeClick() {
  positive_flag = false;
}

async function onSaveClick() {
  await fetch("/save", {
    method: "POST",
  });
  await reset();
}

async function onNextClick() {
  await fetch("/next", {
    method: "POST",
  });
  await reset();
}

async function reset() {
  var colors = points.geometry.attributes.color;
  for (var i = 0; i < origin_colors.count; i++) {
    colors.setXYZ(
      i,
      origin_colors.getX(i),
      origin_colors.getY(i),
      origin_colors.getZ(i)
    );
  }
  colors.needsUpdate = true;
  prompts = [];
  labels = [];
  await fetch("/clear", {
    method: "POST",
  });
}

function bindButtons() {
  var negative_button = document.getElementById("annotate-negative");
  negative_button.onclick = onNegativeClick;
  var positive_button = document.getElementById("annotate-positive");
  positive_button.onclick = onPositiveClick;
  var save_button = document.getElementById("save-result");
  save_button.onclick = onSaveClick;
  var reset_button = document.getElementById("clear-result");
  reset_button.onclick = reset;
  var next_button = document.getElementById("annotate-next");
  next_button.onclick = onNextClick;
}

function main() {
  bindButtons();
}

async function onMouseClick(event) {
  event.preventDefault();
  if (event.target !== renderer.domElement) {
    return;
  }

  if (event.button == 0) {
    positive_flag = true;
  } else if (event.button == 2) {
    positive_flag = false;
  }

  var rect = renderer.domElement.getBoundingClientRect();
  var x = (event.clientX - rect.left) / rect.width;
  var y = (event.clientY - rect.top) / rect.height;

  var mouse = new THREE.Vector2(x * 2 - 1, -(y * 2) + 1);
  raycaster.setFromCamera(mouse, camera);
  var intersects = raycaster.intersectObject(points);

  // Send the annotation to the server
  prompts.push(intersects[0].index);
  labels.push(positive_flag);
  var response = await fetch("/segment", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt_point: intersects[0].point.toArray(),
      prompt_label: positive_flag,
    }),
  });
  var data = await response.json();
  var mask = data.seg;

  // Alpha blend the mask with the point cloud
  var colors = points.geometry.attributes.color;
  for (var i = 0; i < mask.length; i++) {
    var origin = [
      structuredClone(origin_colors.getX(i)),
      structuredClone(origin_colors.getY(i)),
      structuredClone(origin_colors.getZ(i)),
    ];
    if (mask[i] > 0) {
      var new_color = [0, 0, 0];
      new_color[0] = origin[0] * (1 - alpha) + mask_color[0] * alpha;
      new_color[1] = origin[1] * (1 - alpha) + mask_color[1] * alpha;
      new_color[2] = origin[2] * (1 - alpha) + mask_color[2] * alpha;
      colors.setXYZ(i, new_color[0], new_color[1], new_color[2]);
    } else {
      colors.setXYZ(i, origin[0], origin[1], origin[2]);
    }
  }
  colors.needsUpdate = true;

  // Show prompt points
  for (var i = 0; i < prompts.length; i++) {
    if (labels[i] > 0) {
      colors.setXYZ(prompts[i], 1, 0, 0);
    } else {
      colors.setXYZ(prompts[i], 0, 1, 0);
    }
  }
  colors.needsUpdate = true;
}

window.addEventListener("mousedown", () => (drag_flag = false));
window.addEventListener("mousemove", () => (drag_flag = true));
window.addEventListener("mouseup", (event) => {
  if (!drag_flag) {
    onMouseClick(event);
  }
});

main();
