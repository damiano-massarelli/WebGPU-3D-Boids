import { cone, cube } from "./3d-primitives";
import { FreeControlledCamera } from "./camera";
import shader from "./shaders.wgsl";
import * as dat from "dat.gui";

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

    const presentationFormat: GPUTextureFormat = context!.getPreferredFormat(
        adapter!
    );
    console.log(presentationFormat);
    context?.configure({
        device,
        size: presentationSize,
        format: presentationFormat,
    });

    return { device, canvas, context, presentationSize, presentationFormat };
}

export async function run() {
    const gui = new dat.GUI();

    if (!("gpu" in navigator)) {
        return;
    }

    const { device, canvas, context, presentationSize, presentationFormat } =
        await configureCanvas("canvas-wegbpu", false);

    // compute and render shader module
    const shaderModule = device.createShaderModule({
        label: "shader module",
        code: shader,
    });

    // create depth texture
    const depthTexture = device.createTexture({
        size: presentationSize,
        format: "depth24plus-stencil8",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // render pipeline
    const renderPipeline = device.createRenderPipeline({
        label: "render pipeline",
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
    });

    // render pipeline for box
    const renderPipelineBox = device.createRenderPipeline({
        label: "render pipeline box",
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
        },
        depthStencil: {
            format: "depth24plus-stencil8",
            depthWriteEnabled: false,
            depthCompare: "less",
        },
    });

    // render pipeline for outline
    const renderPipelineOutline = device.createRenderPipeline({
        label: "render pipeline outline",
        vertex: {
            module: shaderModule,
            entryPoint: "mainVSOutline",
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
            entryPoint: "mainFSOutline",
            targets: [{ format: presentationFormat }],
        },
        primitive: {
            topology: "triangle-list",
            cullMode: "back",
        },
        depthStencil: {
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
        },
    });

    // Compute pipeline
    const computePipeline = device.createComputePipeline({
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
        rule1Distance: 2.2,
        rule2Distance: 1.2,
        rule3Distance: 1,
        rule1Scale: 0.02,
        rule2Scale: 0.02,
        rule3Scale: 0.075,
        boxWidth: 18,
        boxHeight: 12,
        showOutline: true,
    };
    Object.keys(simParams).forEach((k) => {
        gui.add(simParams, k).onFinishChange(updateSimParams);
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
                simParams.rule1Distance,
                simParams.rule2Distance,
                simParams.rule3Distance,
                simParams.rule1Scale,
                simParams.rule2Scale,
                simParams.rule3Scale,
                simParams.boxWidth,
                simParams.boxHeight,
            ])
        );
    }
    updateSimParams();

    // setup ping-pong buffers for boids position and velocity
    const NUM_PARTICLES = 550;
    const initialParticleData = new Float32Array(NUM_PARTICLES * 8); // x, y, z, vx, vy, vz per particle + padding
    for (let i = 0; i < NUM_PARTICLES; ++i) {
        initialParticleData[8 * i + 0] =
            simParams.boxWidth * (2 * Math.random() - 1); // x // TODO scale by max size
        initialParticleData[8 * i + 1] =
            simParams.boxHeight * (2 * Math.random() - 1); // y
        initialParticleData[8 * i + 2] =
            simParams.boxWidth * (2 * Math.random() - 1); // z
        initialParticleData[8 * i + 3] = 0; // padding
        initialParticleData[8 * i + 4] = 2 * Math.random() - 1; // vx TODO scale by max speed
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
        size: 16 * Float32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
    });

    const renderBindGroup = device.createBindGroup({
        label: "render bind group",
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 3,
                resource: {
                    buffer: cameraBuffer,
                },
            },
        ],
    });

    const renderBindGroupOutline = device.createBindGroup({
        label: "render bind group outline",
        layout: renderPipelineOutline.getBindGroupLayout(0),
        entries: [
            {
                binding: 3,
                resource: {
                    buffer: cameraBuffer,
                },
            },
        ],
    });

    const renderBindGroupBox = device.createBindGroup({
        label: "render bind group box",
        layout: renderPipelineBox.getBindGroupLayout(0),
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

    const camera = new FreeControlledCamera(
        canvas,
        (2 * Math.PI) / 5,
        canvas.clientWidth / canvas.clientHeight
    );
    let t = 0;
    function frame() {
        // update camera uniforms
        device.queue.writeBuffer(
            cameraBuffer,
            0,
            camera.updateAndGetViewProjectionMatrix()
        );

        const commandEncoder = device.createCommandEncoder();
        {
            // COMPUTE BOIDS
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, particleBindGroups[t % 2]);
            passEncoder.dispatch(Math.ceil(NUM_PARTICLES / 64));
            passEncoder.end();
        }
        {
            // BOIDS
            const passEncoder = commandEncoder.beginRenderPass({
                colorAttachments: [
                    {
                        loadOp: "clear",
                        clearValue: [0, 0, 0, 1],
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
            passEncoder.setBindGroup(0, renderBindGroupOutline);
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

            passEncoder.setPipeline(renderPipelineBox);
            passEncoder.setVertexBuffer(0, cubePB);
            passEncoder.setVertexBuffer(1, cubeNB);
            passEncoder.setIndexBuffer(cubeIB, "uint32");
            passEncoder.setBindGroup(0, renderBindGroupBox);
            passEncoder.drawIndexed(cubeIndices.length, 1);
            passEncoder.end();
        }

        device.queue.submit([commandEncoder.finish()]);
        ++t;
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}
