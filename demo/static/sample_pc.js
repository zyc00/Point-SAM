import * as THREE from "three";

export function sample(mesh, sampleSize) {
  if (!mesh.geometry || !mesh.geometry.isBufferGeometry) {
    console.error("Mesh geometry is not available or not a BufferGeometry.");
    return;
  }

  let verts = mesh.geometry.attributes.position;
  let uvs = mesh.geometry.attributes.uv;
  let colors = mesh.geometry.attributes.color;
  let ind = mesh.geometry.index;
  let faceAmount = ind.count / 3;
  let samplingGeometry = new THREE.BufferGeometry();
  let positions = [];
  let sampleColors = [];
  let texture = mesh.material.map; // Assuming mesh.material exists and has a texture map

  for (let i = 0; i < sampleSize; i++) {
    // Randomly select a triangle weighted possibly by face size (left as an exercise)
    let faceIdx = Math.floor(Math.random() * faceAmount) * 3;

    // UVs of the triangle's vertices
    let uvA = new THREE.Vector2().fromBufferAttribute(uvs, ind.array[faceIdx]);
    let uvB = new THREE.Vector2().fromBufferAttribute(
      uvs,
      ind.array[faceIdx + 1]
    );
    let uvC = new THREE.Vector2().fromBufferAttribute(
      uvs,
      ind.array[faceIdx + 2]
    );

    // Get vertex indices
    let aIdx = ind.array[faceIdx];
    let bIdx = ind.array[faceIdx + 1];
    let cIdx = ind.array[faceIdx + 2];

    // Get vertices
    let aVert = new THREE.Vector3().fromBufferAttribute(verts, aIdx);
    let bVert = new THREE.Vector3().fromBufferAttribute(verts, bIdx);
    let cVert = new THREE.Vector3().fromBufferAttribute(verts, cIdx);

    // Barycentric coordinates for a random point
    let baryCoord = randomBarycentric();

    let sampledPosition = aVert
      .multiplyScalar(baryCoord.x)
      .add(bVert.clone().multiplyScalar(baryCoord.y))
      .add(cVert.clone().multiplyScalar(baryCoord.z));

    // Interpolated UV coordinates for the random point
    let uv = uvA
      .clone()
      .multiplyScalar(baryCoord.x)
      .add(uvB.clone().multiplyScalar(baryCoord.y))
      .add(uvC.clone().multiplyScalar(baryCoord.z));

    // Color sampling from the texture
    let color = sampleTexture(texture, uv);

    // Store color of random point
    // TODO: you shold also push sampled position to positions array
    positions.push(sampledPosition.x, sampledPosition.y, sampledPosition.z);
    sampleColors.push(color.r, color.g, color.b);
  }

  // Rest of the code for position sampling is the same...
  // ...

  // Create Buffer Attributes for the geometry
  samplingGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3)
  );
  samplingGeometry.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(sampleColors, 3)
  );

  // Create Points Material
  let material = new THREE.PointsMaterial({ vertexColors: true, size: 0.05 });

  // Create Points from Geometry and Material
  let sampledPoints = new THREE.Points(samplingGeometry, material);
  return sampledPoints;
}

function randomBarycentric() {
  let u = Math.random();
  let v = Math.sqrt(Math.random());
  return {
    x: 1 - v,
    y: v * (1 - u),
    z: v * u,
  };
}

function sampleTexture(texture, uv) {
  let image = texture.image;
  let canvas = document.createElement("canvas");
  let context = canvas.getContext("2d");
  canvas.width = image.width;
  canvas.height = image.height;
  context.drawImage(image, 0, 0, image.width, image.height);

  let x = Math.round(uv.x * image.width);
  let y = Math.round((1 - uv.y) * image.height); // UV origin is bottom left, canvas origin is top left

  let pixelData = context.getImageData(x, y, 1, 1).data;
  let color = new THREE.Color();
  color.r = pixelData[0] / 255;
  color.g = pixelData[1] / 255;
  color.b = pixelData[2] / 255;

  return color;
}
