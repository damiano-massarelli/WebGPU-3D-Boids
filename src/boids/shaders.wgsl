struct Particle {
    pos: vec4<f32>;
    vel: vec4<f32>;
};

struct SimParams {
    deltaT: f32;
    rule1Distance: f32;
    rule2Distance: f32;
    rule3Distance: f32;
    rule1Scale: f32;
    rule2Scale: f32;
    rule3Scale: f32;
    boxWidth: f32;
    boxHeight: f32;
};

struct Particles {
    particles: array<Particle>;
};

struct LightData {
    position: vec4<f32>;       // 0
    cameraPosition: vec4<f32>; // 16
    color: vec4<f32>;          // 32
    ambientIntensity: f32;     // 48
    specularIntensity: f32;    // 52
};

struct Material {
    color: vec4<f32>; // 0
    shininess: f32;   // 16
};

fn computeLight(light: LightData, material: Material, position: vec3<f32>, normal: vec3<f32>) -> vec4<f32> {
    let N: vec3<f32> = normalize(normal.xyz);
    let L: vec3<f32> = normalize(light.position.xyz - position.xyz);
    let V: vec3<f32> = normalize(light.cameraPosition.xyz - position.xyz);
    let H: vec3<f32> = normalize(L + V);
    let kD: f32 = max(dot(N, L), 0.0) + light.ambientIntensity;
    let kS: f32 = light.specularIntensity * pow(max(dot(N, H), 0.0), material.shininess);
    let finalColor = material.color.rgb * light.color.rgb * kD + light.color.rgb * kS;
    return vec4<f32>(finalColor, material.color.a);
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
var<uniform> viewProjectionMatrix: mat4x4<f32>;

@stage(compute)
@workgroup_size(64)
fn mainCS(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let index: u32 = GlobalInvocationID.x;

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
            // push away this boid based on the position 
            // of the neighbours.
            colVel = colVel + (vPos - pos);
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
    
    var d = normalize(vVel);

    for (var j: u32 = 0u; j < 6u; j = j + 1u) {
        let n = planes[j].xyz;
        let o = planes[j].w;
        let dist = abs(dot(vPos, n) - o);
        
        let maxDist = 6.0;
        let minDist = 1.0;

        if (dist < maxDist) {
            var intensity = 1.0 - (dist - minDist) / (maxDist - minDist);
            intensity = clamp(intensity, 0.0, 1.0);
            var directionIntensity = 0.075;
            if (dot(vVel, n) > 0.0) { // reduce repulsion intensity if the boid is going in the direction of the plane normal
                directionIntensity = directionIntensity / 2.5;
            }
            d = normalize(d + directionIntensity * intensity * n);

            if (dist < minDist / 2.0) {
                d = n; // make sure boids do not go outside the box
            }
        }
    }

    // clamp velocity
    vVel = d * clamp(length(vVel), 0.3, 1.0);

    // move the particle
    vPos = vPos + (vVel * params.deltaT);

    // write new calculated data
    particlesB.particles[index].pos = vec4<f32>(vPos, 0.0);
    particlesB.particles[index].vel = vec4<f32>(vVel, 0.0);
}

struct DebugOutput {
    @builtin(position)
    pos: vec4<f32>;

    @location(0)
    col: vec4<f32>;
};

@stage(vertex)
fn mainVS(@location(0) a_particlePos : vec3<f32>,
             @location(1) a_particleVel : vec3<f32>,
             @location(2) a_pos : vec3<f32>,
             @location(3) a_norm: vec3<f32>) -> DebugOutput {
  
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

    var output: DebugOutput;
    let wsPosition = worldMatrix * vec4<f32>(a_pos, 1.0);
    let wsNormal = worldMatrix * vec4<f32>(a_norm, 0.0); // this is ok since there is no scale

    var m: Material;
    m.color = vec4<f32>((a_particleVel + 1.0) / 2.0, 1.0);
    m.shininess = 32.0;

    var l: LightData;
    l.position = vec4<f32>(0.0, 30.0, 0.0, 0.0);
    l.cameraPosition = vec4<f32>(0.0, 0.0, 30.0, 0.0);
    l.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    l.ambientIntensity = 0.5;
    l.specularIntensity = 0.8;

    output.pos = viewProjectionMatrix * wsPosition;
    //output.col = vec4<f32>((a_particleVel + 1.0) / 2.0, 1.0);
    output.col = computeLight(l, m, wsPosition.xyz, wsNormal.xyz);
;
    return output;
}

@stage(fragment)
fn mainFS(@location(0) col: vec4<f32>) -> @location(0) vec4<f32> {
  return col;
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

  return viewProjectionMatrix * worldMatrix * scale * vec4<f32>(a_pos, 1.0);
}

@stage(fragment)
fn mainFSOutline() -> @location(0) vec4<f32> {
    return vec4<f32>(0.68, 0.85, 0.9, 1.0);
}

struct BoxData {
    @builtin(position)
    position: vec4<f32>;

    @location(0)
    normal: vec3<f32>;
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
    out.position = viewProjectionMatrix * scale * vec4<f32>(a_pos, 1.0);
    out.normal = a_norm;
    return out;
}

@stage(fragment)
fn mainFSBox(in: BoxData) -> @location(0) vec4<f32> {
    if (in.normal.y < -0.5) {
        return vec4<f32>(0.15, 0.15, 0.15, 1.0);
    }
    return vec4<f32>(1.0, 1.0, 1.0, 0.1);
}