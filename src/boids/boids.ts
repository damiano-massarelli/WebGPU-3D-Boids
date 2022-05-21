import { cone, cube } from "./3d-primitives";
import { FreeControlledCamera, TurnTableCamera } from "./camera";
import shader from "./shaders.wgsl";
import * as dat from "dat.gui";
import { mat4, vec3, vec4 } from "gl-matrix";

const USE_DEVICE_PIXEL_RATIO = true;

async function configureShadowMap(device: GPUDevice, shadowMapRes: number) {
    const shadowDepthTexture = device.createTexture({
        label: "shadow map texture",
        format: "depth32float",
        size: [shadowMapRes, shadowMapRes, 1],
        usage:
            GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    const shadowDepthTextureView = shadowDepthTexture.createView();
    const sampler = device.createSampler({
        label: "shadow map sampler",
        compare: "less",
        magFilter: "linear",
        minFilter: "linear",
    });

    return {
        shadowMapTextureView: shadowDepthTextureView,
        shadowMapSampler: sampler,
    };
}

async function configureCanvas(canvasId: string, useDevicePixelRatio: boolean) {
    const canvas = document.getElementById(canvasId) as HTMLCanvasElement;

    const context = canvas.getContext("webgpu");

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter!.requestDevice();

    const devicePixelRatio = useDevicePixelRatio
        ? window.devicePixelRatio ?? 1
        : 1;
    const presentationSize = [
        canvas.clientWidth * devicePixelRatio,
        canvas.clientHeight * devicePixelRatio,
    ];

    canvas.width = presentationSize[0];
    canvas.height = presentationSize[1];

    const presentationFormat: GPUTextureFormat =
        navigator.gpu.getPreferredCanvasFormat();
    console.log(presentationFormat);
    context?.configure({
        device,
        size: presentationSize,
        format: presentationFormat,
        compositingAlphaMode: "opaque",
    });

    return { device, canvas, context, presentationSize, presentationFormat };
}

export async function run() {
    const gui = new dat.GUI();

    if (!("gpu" in navigator)) {
        return;
    }

    const { device, canvas, context, presentationSize, presentationFormat } =
        await configureCanvas("canvas-wegbpu", USE_DEVICE_PIXEL_RATIO);

    // compute and render shader module
    const shaderModule = device.createShaderModule({
        label: "shader module",
        code: shader,
    });

    // create depth texture
    let depthTexture = device.createTexture({
        size: presentationSize,
        format: "depth24plus-stencil8",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Background pipeline
    const backgroundPipeline = device.createRenderPipeline({
        layout: "auto",
        label: "background pipeline",
        vertex: {
            module: shaderModule,
            entryPoint: "mainVSBackground",
        },
        fragment: {
            module: shaderModule,
            entryPoint: "mainFSBackground",
            targets: [
                {
                    format: presentationFormat, // output format of the output, that of the swapchain here
                },
            ],
        },
        primitive: {
            topology: "triangle-strip",
        },
    });

    const renderBindGroupLayout = device.createBindGroupLayout({
        label: "render bind group layout",
        entries: [
            {
                // Simulation params
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform",
                },
            },
            {
                // Camera
                binding: 3,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform",
                },
            },
            {
                // Light
                binding: 4,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform",
                },
            },
            {
                // Shadow texture
                binding: 5,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                texture: {
                    sampleType: "depth",
                },
            },
            {
                // Shadow comparison sampler
                binding: 6,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                sampler: {
                    type: "comparison",
                },
            },
        ],
    });

    const renderBindGroupShadowLayout = device.createBindGroupLayout({
        label: "render bind group for shaow casting",
        entries: [
            {
                // Simulation params
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform",
                },
            },
            {
                // Light
                binding: 4,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: {
                    type: "uniform",
                },
            },
        ],
    });

    const boidsRenderPipelineDesc: GPURenderPipelineDescriptor = {
        label: "render pipeline",
        layout: device.createPipelineLayout({
            label: "boid render pipeline layout",
            bindGroupLayouts: [renderBindGroupLayout],
        }),
        vertex: {
            module: shaderModule,
            entryPoint: "mainVS",
            buffers: [
                {
                    arrayStride: 8 * Float32Array.BYTES_PER_ELEMENT, // 6 floats, xyz position and direction
                    stepMode: "instance", // same index for all the vertices of a boid
                    attributes: [
                        {
                            // instance position
                            shaderLocation: 0,
                            offset: 0,
                            format: "float32x4",
                        },
                        {
                            // instance velocity
                            shaderLocation: 1,
                            offset: 4 * Float32Array.BYTES_PER_ELEMENT,
                            format: "float32x4",
                        },
                    ],
                },
                {
                    // vertex positions
                    arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT, // xyz per vertex
                    stepMode: "vertex",
                    attributes: [
                        {
                            shaderLocation: 2,
                            format: "float32x3",
                            offset: 0,
                        },
                    ],
                },
                {
                    // vertex normals
                    arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT, // xyz per vertex
                    stepMode: "vertex",
                    attributes: [
                        {
                            shaderLocation: 3,
                            format: "float32x3",
                            offset: 0,
                        },
                    ],
                },
            ],
        },
        fragment: {
            module: shaderModule,
            entryPoint: "mainFS",
            targets: [{ format: presentationFormat }],
        },
        primitive: {
            topology: "triangle-list",
            cullMode: "back",
        },
        depthStencil: {
            format: "depth24plus-stencil8",
            depthWriteEnabled: true,
            depthCompare: "less",
            stencilFront: {
                compare: "always",
                depthFailOp: "keep",
                failOp: "keep",
                passOp: "replace",
            },
            stencilReadMask: 0xff,
            stencilWriteMask: 0xff,
        },
    };

    // render pipeline
    const renderPipeline = device.createRenderPipeline(boidsRenderPipelineDesc);

    // render pipeline for outline
    boidsRenderPipelineDesc.vertex.entryPoint = "mainVSOutline";
    boidsRenderPipelineDesc.fragment!.entryPoint = "mainFSOutline";
    boidsRenderPipelineDesc.depthStencil = {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "always",
        stencilFront: {
            compare: "not-equal",
            depthFailOp: "keep",
            failOp: "keep",
            passOp: "keep",
        },
        stencilReadMask: 0xff,
        stencilWriteMask: 0x00,
    };
    const renderPipelineOutline = device.createRenderPipeline(
        boidsRenderPipelineDesc
    );

    // render pipeline for boids shadows
    boidsRenderPipelineDesc.vertex.entryPoint = "mainVSShadow";
    boidsRenderPipelineDesc.fragment = undefined;
    boidsRenderPipelineDesc.label = "boids render pipeline shadow";
    boidsRenderPipelineDesc.depthStencil = {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth32float",
    };
    boidsRenderPipelineDesc.layout = device.createPipelineLayout({
        bindGroupLayouts: [renderBindGroupShadowLayout],
    });
    const renderPipelineShadow = device.createRenderPipeline(
        boidsRenderPipelineDesc
    );

    const boxPipelineDescr: GPURenderPipelineDescriptor = {
        label: "render pipeline box",
        layout: device.createPipelineLayout({
            label: "box pipeline layout",
            bindGroupLayouts: [renderBindGroupLayout],
        }),
        vertex: {
            module: shaderModule,
            entryPoint: "mainVSBox",
            buffers: [
                {
                    // vertex positions
                    arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT, // xyz per vertex
                    stepMode: "vertex",
                    attributes: [
                        {
                            shaderLocation: 0,
                            format: "float32x3",
                            offset: 0,
                        },
                    ],
                },
                {
                    // vertex normals
                    arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT, // xyz per vertex
                    stepMode: "vertex",
                    attributes: [
                        {
                            shaderLocation: 1,
                            format: "float32x3",
                            offset: 0,
                        },
                    ],
                },
            ],
        },
        fragment: {
            module: shaderModule,
            entryPoint: "mainFSBox",
            targets: [
                {
                    format: presentationFormat,
                    blend: {
                        color: {
                            operation: "add",
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                        },
                        alpha: {
                            // WebGPU blends the final alpha value
                            // with the page color. We need to make sure the final alpha
                            // value is 1.0 otherwise the final color will be blended with
                            // whatever color the page (behind the canvas) is.
                            operation: "add",
                            srcFactor: "one-minus-dst-alpha",
                            dstFactor: "dst-alpha",
                        },
                    },
                },
            ],
        },
        primitive: {
            topology: "triangle-list",
            cullMode: "front",
        },
        depthStencil: {
            format: "depth24plus-stencil8",
            depthWriteEnabled: false,
            depthCompare: "less",
        },
    };

    // render pipeline for box
    const renderPipelineBoxBack = device.createRenderPipeline(boxPipelineDescr);
    boxPipelineDescr.primitive!.cullMode = "back";
    const renderPipelineBoxFront =
        device.createRenderPipeline(boxPipelineDescr);

    // Compute pipeline
    const computePipeline = device.createComputePipeline({
        layout: "auto",
        label: "compute pipeline",
        compute: {
            module: shaderModule,
            entryPoint: "mainCS",
        },
    });

    // Setup boid geometry
    const {
        positions: conePositions,
        normals: coneNormals,
        indices: coneIndices,
    } = cone(0.4, 20);

    // vertex position buffer
    const conePB = device.createBuffer({
        label: "cone vertex pos buffer",
        size: conePositions.length * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(conePB.getMappedRange()).set(conePositions);
    conePB.unmap();

    // vertex normal buffer
    const coneNB = device.createBuffer({
        label: "cone vertex normal buffer",
        size: coneNormals.length * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(coneNB.getMappedRange()).set(coneNormals);
    coneNB.unmap();

    // index buffer
    const coneIB = device.createBuffer({
        label: "cone index buffer",
        size: coneIndices.length * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
    });
    new Uint32Array(coneIB.getMappedRange()).set(coneIndices);
    coneIB.unmap();

    // Setup box geometry
    const {
        positions: cubePositions,
        normals: cubeNormals,
        indices: cubeIndices,
    } = cube();

    // vertex position buffer
    const cubePB = device.createBuffer({
        label: "cone vertex pos buffer",
        size: cubePositions.length * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(cubePB.getMappedRange()).set(cubePositions);
    cubePB.unmap();

    // vertex normal buffer
    const cubeNB = device.createBuffer({
        label: "cone vertex normal buffer",
        size: cubeNormals.length * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(cubeNB.getMappedRange()).set(cubeNormals);
    cubeNB.unmap();

    // index buffer
    const cubeIB = device.createBuffer({
        label: "cone index buffer",
        size: cubeIndices.length * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
    });
    new Uint32Array(cubeIB.getMappedRange()).set(cubeIndices);
    cubeIB.unmap();

    // simulation parameters
    const simParams = {
        deltaT: 0.07,
        cohesionDistance: 2,
        separationDistance: 1.2,
        alignmentDistance: 1.15,
        cohesionScale: 0.02,
        separationScale: 0.02,
        alignmentScale: 0.07,
        boxWidth: 20,
        boxHeight: 12,
        showOutline: false,
        freeCamera: false,

        meta: {
            deltaT: {
                min: 0,
                max: 0.15,
            },
            cohesionDistance: {
                min: 0,
                max: 3,
            },
            separationDistance: {
                min: 0,
                max: 3,
            },
            alignmentDistance: {
                min: 0,
                max: 3,
            },
            cohesionScale: {
                min: 0,
                max: 1,
            },
            separationScale: {
                min: 0,
                max: 1,
            },
            alignmentScale: {
                min: 0,
                max: 1,
            },
            boxWidth: {
                max: 20,
                min: 8,
            },
            boxHeight: {
                max: 14,
                min: 4,
            },
            freeCamera: {
                toolTip:
                    "Once selected click on the viewport to control the camera with WASD + mouse. Esc to exit",
            },
        } as {
            [key: string]:
                | { min?: number; max?: number; toolTip?: string }
                | undefined;
        },
    };
    Object.keys(simParams).forEach((k) => {
        if (k !== "meta") {
            const controller = gui.add(
                simParams,
                k,
                simParams.meta[k]?.min,
                simParams.meta[k]?.max
            );
            controller.onChange(updateSimParams);

            // very cheap way of creating tooltips
            controller.domElement.parentElement!.setAttribute(
                "title",
                simParams.meta[k]?.toolTip ?? ""
            );
        }
    });

    const simParamBufferSize = 9 * Float32Array.BYTES_PER_ELEMENT;
    const simParamBuffer = device.createBuffer({
        label: "simulation params buffer",
        size: simParamBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, // It is totally legit to pass uniform buffers to compute shader
    });

    function updateSimParams() {
        device.queue.writeBuffer(
            simParamBuffer,
            0,
            new Float32Array([
                simParams.deltaT,
                simParams.cohesionDistance,
                simParams.separationDistance,
                simParams.alignmentDistance,
                simParams.cohesionScale,
                simParams.separationScale,
                simParams.alignmentScale,
                simParams.boxWidth,
                simParams.boxHeight,
            ])
        );
    }
    updateSimParams();

    // Light data
    const lightPosition = vec4.fromValues(
        simParams.boxWidth / 2,
        simParams.boxHeight * 1.5,
        simParams.boxWidth / 1.5,
        0
    );
    const lightDirection = vec4.create();
    vec4.negate(lightDirection, lightPosition);
    vec4.normalize(lightDirection, lightDirection);
    const lightViewProj = mat4.create();
    {
        const viewMatrix = mat4.create();
        mat4.lookAt(
            viewMatrix,
            lightPosition as vec3,
            vec3.fromValues(0, 0, 0),
            vec3.fromValues(0, 1, 0)
        );
        const lightProjectionMatrix = mat4.create();
        mat4.orthoZO(
            lightProjectionMatrix,
            -1.75 * simParams.boxWidth,
            1.75 * simParams.boxWidth,
            -1.75 * simParams.boxWidth,
            1.75 * simParams.boxWidth,
            0,
            5 * simParams.boxHeight
        );
        mat4.mul(lightViewProj, lightProjectionMatrix, viewMatrix);
    }
    const lightData = {
        viewProjection: lightViewProj,
        position: lightPosition,
        direction: lightDirection,
        color: vec4.fromValues(0.8, 0.8, 0.8, 1.0),
        ambientIntensity: 0.5,
    };
    const lightDataBuffer = device.createBuffer({
        label: "light data buffer",
        size: (16 + 4 * 3 + 1) * Float32Array.BYTES_PER_ELEMENT, // 1 4x4 matrix + 3 vec4 + 1 float
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(lightDataBuffer.getMappedRange()).set([
        ...lightData.viewProjection,
        ...lightData.position,
        ...lightData.direction,
        ...lightData.color,
        lightData.ambientIntensity,
    ]);
    lightDataBuffer.unmap();

    // setup ping-pong buffers for boids position and velocity
    const NUM_PARTICLES = 550;
    const initialParticleData = new Float32Array(NUM_PARTICLES * 8); // x, y, z, vx, vy, vz per particle + padding
    for (let i = 0; i < NUM_PARTICLES; ++i) {
        initialParticleData[8 * i + 0] =
            simParams.boxWidth * (2 * Math.random() - 1); // x
        initialParticleData[8 * i + 1] =
            simParams.boxHeight * (2 * Math.random() - 1); // y
        initialParticleData[8 * i + 2] =
            simParams.boxWidth * (2 * Math.random() - 1); // z
        initialParticleData[8 * i + 3] = 0; // padding
        initialParticleData[8 * i + 4] = 2 * Math.random() - 1; // vx
        initialParticleData[8 * i + 5] = 2 * Math.random() - 1; // vy
        initialParticleData[8 * i + 6] = 2 * Math.random() - 1; // vz
        initialParticleData[8 * i + 7] = 0; // padding
    }

    const particleBuffers: GPUBuffer[] = new Array(2);
    const particleBindGroups: GPUBindGroup[] = new Array(2);
    for (let i = 0; i < 2; ++i) {
        particleBuffers[i] = device.createBuffer({
            label: `particle buffer ${i}`,
            size: initialParticleData.byteLength,
            // Vertex when reading for rendering, storage when reading / writing in compute
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Float32Array(particleBuffers[i].getMappedRange()).set(
            initialParticleData
        );
        particleBuffers[i].unmap();
    }

    const cameraBuffer = device.createBuffer({
        label: "cameraBuffer",
        size: (16 + 4 + 3 + 1) * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
    });

    const { shadowMapTextureView, shadowMapSampler } = await configureShadowMap(
        device,
        1024
    );
    const renderBindGroup = device.createBindGroup({
        label: "render bind group",
        layout: renderBindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: simParamBuffer,
                },
            },
            {
                binding: 3,
                resource: {
                    buffer: cameraBuffer,
                },
            },
            {
                binding: 4,
                resource: {
                    buffer: lightDataBuffer,
                },
            },
            {
                binding: 5,
                resource: shadowMapTextureView,
            },
            {
                binding: 6,
                resource: shadowMapSampler,
            },
        ],
    });

    const renderBindGroupShadow = device.createBindGroup({
        label: "render bind group shadow",
        layout: renderBindGroupShadowLayout,
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: simParamBuffer,
                },
            },
            {
                binding: 4,
                resource: {
                    buffer: lightDataBuffer,
                },
            },
        ],
    });

    const renderBindGroupBackground = device.createBindGroup({
        label: "render bind ground bg",
        layout: backgroundPipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 3,
                resource: {
                    buffer: cameraBuffer,
                },
            },
        ],
    });

    for (let i = 0; i < 2; ++i) {
        particleBindGroups[i] = device.createBindGroup({
            label: `particle bind group ${i}`,
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: simParamBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: particleBuffers[i],
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: particleBuffers[(i + 1) % 2],
                    },
                },
            ],
        });
    }

    const freeCamera = new FreeControlledCamera(
        canvas,
        (2 * Math.PI) / 5,
        presentationSize[0] / presentationSize[1]
    );

    const turnCamera = new TurnTableCamera(
        (2 * Math.PI) / 5,
        presentationSize[0] / presentationSize[1]
    );
    turnCamera.activate();
    turnCamera.rotationPivot = vec3.fromValues(
        0,
        simParams.boxHeight * 1.25,
        0
    );
    turnCamera.rotatationRadius =
        Math.max(simParams.boxWidth, simParams.boxHeight) * 2;

    turnCamera.rotationSpeed = 0.005;

    let camera: TurnTableCamera | FreeControlledCamera = turnCamera;

    let t = 0;
    let currentWidth = canvas.clientWidth;
    let currentHeight = canvas.clientHeight;
    function frame() {
        if (
            currentWidth !== canvas.clientWidth ||
            currentHeight != canvas.clientHeight
        ) {
            const devicePixelRatio = USE_DEVICE_PIXEL_RATIO
                ? window.devicePixelRatio ?? 1
                : 1;
            const presentationSize = [
                canvas.clientWidth * devicePixelRatio,
                canvas.clientHeight * devicePixelRatio,
            ];
            context?.configure({
                device,
                size: presentationSize,
                format: presentationFormat,
                compositingAlphaMode: "opaque",
            });
            depthTexture.destroy();
            depthTexture = device.createTexture({
                size: presentationSize,
                format: "depth24plus-stencil8",
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
            currentWidth = canvas.clientWidth;
            currentHeight = canvas.clientHeight;

            turnCamera.aspectRatio = presentationSize[0] / presentationSize[1];
            freeCamera.aspectRatio = presentationSize[0] / presentationSize[1];
        }

        turnCamera.lookAt = vec3.fromValues(0, -simParams.boxHeight / 1.5, 0);
        if (simParams.freeCamera) {
            if (camera !== freeCamera) {
                freeCamera.copyTransform(turnCamera);
            }
            camera.deactivate();
            camera = freeCamera;
            camera.activate();
        } else {
            camera.deactivate();
            camera = turnCamera;
            camera.activate();
        }

        // update camera uniforms
        device.queue.writeBuffer(
            cameraBuffer,
            0,
            new Float32Array([
                ...camera.updateAndGetViewProjectionMatrix(),
                ...camera.position,
                0, // padding
                ...camera.rotation,
                camera.fovY,
            ])
        );

        const commandEncoder = device.createCommandEncoder();
        {
            // COMPUTE BOIDS
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, particleBindGroups[t % 2]);
            passEncoder.dispatchWorkgroups(Math.ceil(NUM_PARTICLES / 64));
            passEncoder.end();
        }
        {
            // BOIDS shadow
            const passEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [],
                depthStencilAttachment: {
                    view: shadowMapTextureView,
                    depthClearValue: 1,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                },
            });

            passEncoder.setPipeline(renderPipelineShadow);
            passEncoder.setVertexBuffer(0, particleBuffers[(t + 1) % 2]);
            passEncoder.setVertexBuffer(1, conePB);
            passEncoder.setVertexBuffer(2, coneNB);
            passEncoder.setIndexBuffer(coneIB, "uint32");
            passEncoder.setBindGroup(0, renderBindGroupShadow);
            passEncoder.drawIndexed(coneIndices.length, NUM_PARTICLES);
            passEncoder.end();
        }
        {
            // BACKGROUND
            const passEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [
                    {
                        loadOp: "clear",
                        clearValue: [0, 0, 0, 1],
                        storeOp: "store",
                        view: context!.getCurrentTexture().createView(),
                    },
                ],
            });
            passEncoder.setPipeline(backgroundPipeline);
            passEncoder.setBindGroup(0, renderBindGroupBackground);
            passEncoder.draw(4);
            passEncoder.end();
        }
        {
            // BOIDS
            const passEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [
                    {
                        loadOp: "load",
                        //clearValue: [0.08, 0.1, 0.54, 1],
                        storeOp: "store",
                        view: context!.getCurrentTexture().createView(),
                    },
                ],
                depthStencilAttachment: {
                    view: depthTexture.createView(),
                    depthClearValue: 1,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                    stencilClearValue: 0,
                    stencilStoreOp: "store",
                    stencilLoadOp: "clear",
                },
            });

            // (t + 1) % 2. at t = 0 the compute buffer writes to
            // the second buffer, see the first element in
            // particleBindGroups
            passEncoder.setPipeline(renderPipeline);
            passEncoder.setVertexBuffer(0, particleBuffers[(t + 1) % 2]);
            passEncoder.setVertexBuffer(1, conePB);
            passEncoder.setVertexBuffer(2, coneNB);
            passEncoder.setIndexBuffer(coneIB, "uint32");
            passEncoder.setBindGroup(0, renderBindGroup);
            passEncoder.setStencilReference(0x01);
            passEncoder.drawIndexed(coneIndices.length, NUM_PARTICLES);
            passEncoder.end();
        }
        // OUTLINE
        if (simParams.showOutline) {
            const passEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [
                    {
                        loadOp: "load",
                        clearValue: [0, 0, 0, 1],
                        storeOp: "store",
                        view: context!.getCurrentTexture().createView(),
                    },
                ],
                depthStencilAttachment: {
                    view: depthTexture.createView(),
                    depthClearValue: 1,
                    depthLoadOp: "load",
                    depthStoreOp: "store",
                    stencilClearValue: 0,
                    stencilStoreOp: "store",
                    stencilLoadOp: "load",
                },
            });

            // (t + 1) % 2. at t = 0 the compute buffer writes to
            // the second buffer, see the first element in
            // particleBindGroups
            passEncoder.setPipeline(renderPipelineOutline);
            passEncoder.setVertexBuffer(0, particleBuffers[(t + 1) % 2]);
            passEncoder.setVertexBuffer(1, conePB);
            passEncoder.setVertexBuffer(2, coneNB);
            passEncoder.setIndexBuffer(coneIB, "uint32");
            passEncoder.setBindGroup(0, renderBindGroup);
            passEncoder.setStencilReference(0x01);
            passEncoder.drawIndexed(coneIndices.length, NUM_PARTICLES);
            passEncoder.end();
        }
        {
            // BOX
            const passEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [
                    {
                        loadOp: "load",
                        clearValue: [0, 0, 0, 1],
                        storeOp: "store",
                        view: context!.getCurrentTexture().createView(),
                    },
                ],
                depthStencilAttachment: {
                    view: depthTexture.createView(),
                    depthClearValue: 1,
                    depthLoadOp: "load",
                    depthStoreOp: "store",
                    stencilClearValue: 0,
                    stencilStoreOp: "store",
                    stencilLoadOp: "clear",
                },
            });

            passEncoder.setPipeline(renderPipelineBoxBack);
            passEncoder.setBindGroup(0, renderBindGroup);
            passEncoder.setVertexBuffer(0, cubePB);
            passEncoder.setVertexBuffer(1, cubeNB);
            passEncoder.setIndexBuffer(cubeIB, "uint32");
            passEncoder.drawIndexed(cubeIndices.length, 1);

            passEncoder.setPipeline(renderPipelineBoxFront);
            passEncoder.drawIndexed(cubeIndices.length, 1);

            passEncoder.end();
        }

        device.queue.submit([commandEncoder.finish()]);
        ++t;
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}
