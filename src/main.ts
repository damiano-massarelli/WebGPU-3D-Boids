import { run } from "./boids/boids";

if (navigator.gpu) {
    run();
} else {
    document.getElementById("webgpu-available")!.innerText =
        "Looks like webgpu is not available :(";
}
