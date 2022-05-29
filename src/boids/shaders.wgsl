struct Particle {
    pos: vec4<f32>,
    vel: vec4<f32>,
};

struct SimParams {
    deltaT: f32,
    rule1Distance: f32,
    rule2Distance: f32,
    rule3Distance: f32,
    rule1Scale: f32,
    rule2Scale: f32,
    rule3Scale: f32,
    boxWidth: f32,
    boxHeight: f32,
};

struct Particles {
    particles: array<Particle>,
};

struct CameraData {
    viewProjectionMatrix: mat4x4<f32>, // 0
    position: vec4<f32>,               // 64
    rotation: vec3<f32>,               // 80
    fovYRad: f32,                      // 92
};

struct LightData {
    viewProjectionMatrix: mat4x4<f32>, // 0
    position: vec4<f32>,               // 64
    direction: vec4<f32>,              // 80
    color: vec4<f32>,                  // 96
    ambientIntensity: f32,             // 112
};

struct Material {
    color: vec4<f32>,        // 0
    shininess: f32,          // 16
    specularIntensity: f32,  // 20
};

// from https://www.shadertoy.com/view/4djSRW
fn hash13(input: vec3<f32>) -> f32 {
	var p3  = fract(input * .1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

fn computeLight(light: LightData, material: Material, cameraPosition: vec3<f32>, position: vec3<f32>, normal: vec3<f32>, visibility: f32, shadowAttenuation: f32) -> vec4<f32> {
    let shadow = mix(shadowAttenuation, 1.0, visibility);
    let N: vec3<f32> = normalize(normal.xyz);
    let L: vec3<f32> = normalize(-light.direction.xyz);
    let V: vec3<f32> = normalize(cameraPosition.xyz - position.xyz);
    let H: vec3<f32> = normalize(L + V);
    let NdotL = max(dot(N, L), 0.0);
    let kD: f32 = shadow * NdotL + light.ambientIntensity;
    var kSEnabled = 0.0;

    let R = reflect(-L, N);
    if (NdotL > 0.0 && dot(V, N) > 0.0) {
        kSEnabled = 1.0;
    }
    var alpha = material.color.a;
    let kS: f32 = shadow * kSEnabled * material.specularIntensity * pow(max(dot(H, N), 0.0), material.shininess);
    alpha = mix(alpha, 1.0, kS);
    let finalColor = material.color.rgb * light.color.rgb * kD + light.color.rgb * kS;
    let noise = hash13(position);

    // also add some noise/dither to avoid banding artifacts 
    return vec4<f32>(finalColor, alpha) + mix(-0.5/255.0, 0.5/255.0, noise);
};

@group(0)
@binding(0)
var<uniform> params: SimParams;

@group(0)
@binding(1)
var<storage, read> particlesA: Particles;

@group(0)
@binding(2)
var<storage, read_write> particlesB: Particles;

@group(0)
@binding(3)
var<uniform> cameraData: CameraData;

@group(0)
@binding(4)
var<uniform> lightData: LightData;

@group(0)
@binding(5) 
var shadowMap: texture_depth_2d;

@group(0) 
@binding(6) 
var shadowSampler: sampler_comparison;

@stage(compute)
@workgroup_size(64)
fn mainCS(@builtin(global_invocation_id) globalInvocationID: vec3<u32>) {
    let index: u32 = globalInvocationID.x;

    if (index >= arrayLength(&particlesA.particles)) {
        return;
    }

    // normalX, normalY, normalZ, distance from origin
    let planes = array<vec4<f32>, 6>(
        vec4<f32>( 1.0,  0.0,  0.0, -params.boxWidth), // left
        vec4<f32>(-1.0,  0.0,  0.0, -params.boxWidth), // right
        vec4<f32>( 0.0, -1.0,  0.0, -params.boxHeight), // top
        vec4<f32>( 0.0,  1.0,  0.0, -params.boxHeight), // bottom
        vec4<f32>( 0.0,  0.0, -1.0, -params.boxWidth), // front
        vec4<f32>( 0.0,  0.0,  1.0, -params.boxWidth), // back
    );

    var vPos: vec3<f32> = particlesA.particles[index].pos.xyz;
    var vVel: vec3<f32> = particlesA.particles[index].vel.xyz;

    var cMass = vec3<f32>(0.0, 0.0, 0.0);
    var cVel = vec3<f32>(0.0, 0.0, 0.0);
    var colVel = vec3<f32>(0.0, 0.0, 0.0);
    var cMassCount: u32 = 0u;
    var cVelCount: u32 = 0u;
    var pos: vec3<f32>;
    var vel: vec3<f32>;

    for (var i: u32 = 0u; i < arrayLength(&particlesA.particles); i = i + 1u) {

        if (i == index) {
            continue;
        }

        pos = particlesA.particles[i].pos.xyz;
        vel = particlesA.particles[i].vel.xyz;

        // rule 1: cohesion, steer towards center of mass of
        // local flockmates
        if (distance(pos, vPos) < params.rule1Distance) {
            cMass = cMass + pos;
            cMassCount = cMassCount + 1u;
        }

        // rule2: separation, steer to avoid crowding local flockmates
        if (distance(pos, vPos) < params.rule2Distance) {
            // additional push intensity to avoid compenetration
            let additionalIntensity = mix(4.0, 1.0, distance(pos, vPos) / params.rule2Distance);
            // push away this boid based on the position 
            // of the neighbours.
            colVel = colVel + additionalIntensity * (vPos - pos);
        }

        // rule3: alignment, steer towards the average heading of local flockmates
        if (distance(pos, vPos) < params.rule3Distance) {
            cVel = cVel + vel;
            cVelCount = cVelCount + 1u;
        }
    }

    if (cMassCount > 0u) {
        let temp = f32(cMassCount);
        // this will be added to the particle's vel.
        // steers the particle towards the center of mass
        cMass = (cMass / vec3<f32>(temp, temp, temp)) - vPos;
    }
    if (cVelCount > 0u) {
        let temp = f32(cVelCount);
        cVel = cVel / vec3<f32>(temp, temp, temp);
    }

    vVel = vVel + (cMass * params.rule1Scale) + (colVel * params.rule2Scale) +
        (cVel * params.rule3Scale);
    
    // fishy "random"
    //let offset = 0.1 * vec3<f32>(cos(vPos.x * 5.0 + f32(index) / 10.), sin(vPos.y * 5.0 + f32(index) / 10.), cos(vPos.z * 5.0 + f32(index) / 10.));
    //vVel = vVel + offset * 0.1;
    
    var d = normalize(vVel);

    for (var j: u32 = 0u; j < 6u; j = j + 1u) {
        let n = planes[j].xyz;
        let o = planes[j].w;
        let dist = abs(dot(vPos, n) - o);
        
        let maxDist = 6.0;
        let minDist = 1.0;

        if (dist < maxDist) {
            var intensity = 1.0 - (dist - minDist) / (maxDist - minDist);
            intensity = pow(clamp(intensity, 0.0, 1.0), 0.5);
            var directionIntensity = 0.1;
            if (dot(vVel, n) > 0.0) { // reduce repulsion intensity if the boid is going in the direction of the plane normal
                directionIntensity = directionIntensity / 2.5;
            }
            if (dist < minDist) {
                // boids are too close to the plane, increase repulsion intensity
                directionIntensity = directionIntensity * 10.0 * (minDist - dist) / minDist;
            }

            d = normalize(d + directionIntensity * intensity * n);
            
            if (dist <= 0.1) {
                d = n; // make sure boids do not go outside the box
            }
        }
    }

    // clamp velocity
    vVel = d * clamp(length(vVel), 0.3, 1.0);

    // move the particle
    vPos = vPos + (vVel * params.deltaT);
    vPos = clamp(vPos, vec3<f32>(-params.boxWidth, -params.boxHeight, -params.boxWidth), vec3<f32>(params.boxWidth, params.boxHeight, params.boxWidth));

    // write new calculated data
    particlesB.particles[index].pos = vec4<f32>(vPos, 0.0);
    particlesB.particles[index].vel = vec4<f32>(vVel, 0.0);
}

struct VSOutBoids {
    @builtin(position)
    pos: vec4<f32>,

    @location(0)
    wsPos: vec4<f32>,

    @location(1)
    wsNormal: vec4<f32>,

    @location(2)
    col: vec4<f32>,

    // xy texel position, z: depth from light
    @location(3)
    shadowMapCoords: vec3<f32>,
};

fn getBoidWorldMatrix(position: vec3<f32>, velocity: vec3<f32>) -> mat4x4<f32> {
    // y points towards velocity
    let up = normalize(velocity);
    let right = normalize(cross(up, vec3<f32>(0.0, 0.0, 1.0)));
    let forward = normalize(cross(right, up));

    let worldMatrix = mat4x4<f32>(
        vec4<f32>(right, 0.0),
        vec4<f32>(up, 0.0),
        vec4<f32>(forward, 0.0),
        vec4<f32>(position, 1.0)
    );

    return worldMatrix;
}

fn getVisibility(shadowMapCoords: vec3<f32>) -> f32 {
    let pcfResolution = 1;

    var visibility: f32 = 0.0;
    let offset = 1.0 / vec2<f32>(textureDimensions(shadowMap));
    for (var i = -pcfResolution; i <= pcfResolution; i = i + 1) {
        for (var j = -pcfResolution; j <= pcfResolution; j = j + 1) {
            visibility = visibility + textureSampleCompare(
                shadowMap,
                shadowSampler,
                shadowMapCoords.xy + vec2<f32>(f32(i), f32(j)) * offset, shadowMapCoords.z - 0.007
            );
        }
    }

    if (all(shadowMapCoords.xy <= vec2<f32>(1.0, 1.0)) && all(shadowMapCoords.xy >= vec2<f32>(0.0, 0.0))) {
        visibility = visibility / f32((pcfResolution + pcfResolution + 1) * (pcfResolution + pcfResolution + 1));
    } else {
        visibility = 1.0;
    }

    return visibility;
}

@stage(vertex)
fn mainVS(@location(0) a_particlePos : vec3<f32>,
             @location(1) a_particleVel : vec3<f32>,
             @location(2) a_pos : vec3<f32>,
             @location(3) a_norm: vec3<f32>) -> VSOutBoids {

    let worldMatrix = getBoidWorldMatrix(a_particlePos, a_particleVel);

    var output: VSOutBoids;
    let wsPosition = worldMatrix * vec4<f32>(a_pos, 1.0);
    let wsNormal = worldMatrix * vec4<f32>(a_norm, 0.0); // this is ok since there is no scale

    var shadowProjection = lightData.viewProjectionMatrix * wsPosition;
    shadowProjection = shadowProjection / shadowProjection.w;

    output.shadowMapCoords = vec3<f32>(shadowProjection.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5), shadowProjection.z);

    output.pos = cameraData.viewProjectionMatrix * wsPosition;
    output.wsPos = wsPosition;
    output.wsNormal = wsNormal;
    output.col = vec4<f32>((a_particleVel + 1.0) / 2.0, 1.0);
    return output;
}

@stage(vertex)
fn mainVSShadow(@location(0) a_particlePos : vec3<f32>,
             @location(1) a_particleVel : vec3<f32>,
             @location(2) a_pos : vec3<f32>,
             @location(3) a_norm: vec3<f32>) -> @builtin(position) vec4<f32> {
    
    // y points towards velocity
    let up = normalize(a_particleVel);
    let right = normalize(cross(up, vec3<f32>(0.0, 0.0, 1.0)));
    let forward = normalize(cross(right, up));

    let worldMatrix = getBoidWorldMatrix(a_particlePos, a_particleVel);

    let wsPosition = worldMatrix * vec4<f32>(a_pos, 1.0);

    return lightData.viewProjectionMatrix * wsPosition;
}

@stage(fragment)
fn mainFS(in: VSOutBoids) -> @location(0) vec4<f32> {
    var material: Material;
    material.color = in.col;
    material.shininess = 12.0;
    material.specularIntensity = 0.3;
    return computeLight(lightData, material, cameraData.position.xyz, in.wsPos.xyz, in.wsNormal.xyz, getVisibility(in.shadowMapCoords), 0.0);
}

@stage(vertex)
fn mainVSOutline(@location(0) a_particlePos : vec3<f32>,
             @location(1) a_particleVel : vec3<f32>,
             @location(2) a_pos : vec3<f32>,
             @location(3) a_norm: vec3<f32>) -> @builtin(position) vec4<f32> {
  
  // y points towards velocity
  let up = normalize(a_particleVel);
  let right = normalize(cross(up, vec3<f32>(0.0, 0.0, 1.0)));
  let forward = normalize(cross(right, up));

  let worldMatrix = mat4x4<f32>(
      vec4<f32>(right, 0.0),
      vec4<f32>(up, 0.0),
      vec4<f32>(forward, 0.0),
      vec4<f32>(a_particlePos, 1.0)
  );

  // tweak scale on the y axis to make the cone outline
  // look better
  var sy: f32 = 1.35;
  if (a_pos.y > 0.0)
  {
      sy = 2.0;
  }

  let scale = mat4x4<f32>(
      vec4<f32>(1.5, 0.0, 0.0, 0.0),
      vec4<f32>(0.0, sy, 0.0, 0.0),
      vec4<f32>(0.0, 0.0, 1.5, 0.0),
      vec4<f32>(0.0, 0.0, 0.0, 1.0)
  );

  return cameraData.viewProjectionMatrix * worldMatrix * scale * vec4<f32>(a_pos, 1.0);
}

@stage(fragment)
fn mainFSOutline() -> @location(0) vec4<f32> {
    return vec4<f32>(0.68, 0.85, 0.9, 1.0);
}

struct BoxData {
    @builtin(position)
    position: vec4<f32>,

    @location(0)
    wsPos: vec4<f32>,

    @location(1)
    wsNormal: vec3<f32>,

    // xy texel position, z: depth from light
    @location(3)
    shadowMapCoords: vec3<f32>,
};

@stage(vertex)
fn mainVSBox(@location(0) a_pos : vec3<f32>,
             @location(1) a_norm: vec3<f32>) -> BoxData {
    let scale = mat4x4<f32>(
        params.boxWidth, 0.0,            0.0,            0.0,
        0.0,            params.boxHeight, 0.0,            0.0,
        0.0,            0.0,            params.boxWidth, 0.0,
        0.0,            0.0,            0.0,            1.0
    );
 
    var out: BoxData;
    out.wsPos = scale * vec4<f32>(a_pos, 1.0);
    var shadowProjection = lightData.viewProjectionMatrix * out.wsPos;

    out.shadowMapCoords = vec3<f32>(shadowProjection.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5), shadowProjection.z);
    out.position = cameraData.viewProjectionMatrix * out.wsPos;
    out.wsNormal = a_norm;
    return out;
}

@stage(fragment)
fn mainFSBox(in: BoxData, @builtin(front_facing) frontFacing: bool) -> @location(0) vec4<f32> {
    var material: Material;
    material.color = vec4<f32>(.8, .8, .8, 0.1);
    material.specularIntensity = 1.;
    var wsNormal = in.wsNormal;

    var shadowAttenuation = 0.5;
    // use a different color for the cube base
    if (in.wsNormal.y < -0.5) {
        material.color = vec4<f32>(0.15, 0.15, 0.15, 1.0);
        material.specularIntensity = 0.1;

        if (frontFacing) {
            shadowAttenuation = 1.0; // hide shadow when looking at the base from below
        }
        else {
            wsNormal = -wsNormal;
        }
    }
    else {
        // make sure the normal always points towards the light source
        if (dot(-lightData.direction.xyz, in.wsNormal) < 0.0) {
            wsNormal = -wsNormal;
        }
    }

    material.shininess = 10.0;
    return computeLight(lightData, material, cameraData.position.xyz, in.wsPos.xyz, wsNormal, getVisibility(in.shadowMapCoords), shadowAttenuation);
}

struct BackgroundData {
    @builtin(position)
    position: vec4<f32>,

    @location(0)
    color: vec4<f32>,
};

@stage(vertex)
fn mainVSBackground(@builtin(vertex_index) vertexIndex: u32) -> BackgroundData {
    var screenQuad = array<vec2<f32>, 4>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>(1.0, -1.0),
            vec2<f32>(-1.0, 1.0),
            vec2<f32>(1.0, 1.0));

    var out: BackgroundData;
    out.position = vec4<f32>(screenQuad[vertexIndex], 0.0, 1.0);

    let fovYDeg = cameraData.fovYRad * 180.0 / 3.14; 
    let orbitPitchBlend = sin((cameraData.rotation.x + (out.position.y * fovYDeg/2.0)) * 3.14 / 180.0) * 0.5 + 0.5;
    out.color = mix(vec4<f32>(0.08, 0.1, 0.54, 1.0), vec4<f32>(0.16, 0.24, 0.74, 1.0), orbitPitchBlend);
    return out;
}

@stage(fragment)
fn mainFSBackground(in: BackgroundData) -> @location(0) vec4<f32> {
    return in.color;
}