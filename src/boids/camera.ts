import { mat3, mat4, quat, vec3 } from "gl-matrix";

export class Camera {
    private rotationDeg: [number, number, number];
    private rotationQuat: quat;
    private projectionMatrix: mat4;

    position: vec3;

    constructor(
        fovY: number,
        aspectRatio: number,
        near: number = 0.1,
        far: number = 1000
    ) {
        this.position = [0, 0, 0];
        this.rotationDeg = [0, 0, 0];
        this.rotationQuat = quat.create();
        this.projectionMatrix = mat4.create();
        mat4.perspective(this.projectionMatrix, fovY, aspectRatio, near, far);
    }

    get rotation() {
        return [...this.rotationDeg];
    }

    set rotation(rotationDeg: [number, number, number]) {
        this.rotationDeg = [...rotationDeg];
        quat.fromEuler(this.rotationQuat, ...rotationDeg);
    }

    getUpRightMatrix(): mat3 {
        const result = mat3.create();
        mat3.fromQuat(result, this.rotationQuat);
        return result;
    }

    getViewMatrix(): mat4 {
        const result = mat4.create();
        mat4.fromQuat(result, this.rotationQuat);
        mat4.transpose(result, result);
        const negPosition = vec3.create();
        vec3.negate(negPosition, this.position);
        mat4.translate(result, result, negPosition);
        return result;
    }

    getViewProjectionMatrix(): mat4 {
        const result = mat4.create();
        mat4.mul(result, this.projectionMatrix, this.getViewMatrix());
        return result;
    }
}

export class FreeControlledCamera extends Camera {
    moveSpeed: number = 0.1;
    rotationSpeed: number = 0.1;
    rotateOnlyIfFocussed: boolean = true;

    private readonly state = { forward: 0, right: 0 };
    private hasFocus: boolean = false;

    constructor(
        canvas: HTMLCanvasElement,
        fovY: number,
        aspectRatio: number,
        near: number = 0.1,
        far: number = 1000
    ) {
        super(fovY, aspectRatio, near, far);

        canvas.addEventListener(
            "click",
            () => {
                canvas.requestPointerLock();
            },
            false
        );

        canvas.setAttribute("tabindex", "0");

        const handler = (event: KeyboardEvent) => {
            if (event.repeat) {
                return;
            }

            const dir = event.type === "keydown" ? 1 : -1;
            if (event.key === "w" && dir * this.state.forward >= 0) {
                this.state.forward -= dir * this.moveSpeed;
            }
            if (event.key === "s" && dir * this.state.forward <= 0) {
                this.state.forward += dir * this.moveSpeed;
            }
            if (event.key === "d" && dir * this.state.right <= 0) {
                this.state.right += dir * this.moveSpeed;
            }
            if (event.key === "a" && dir * this.state.right >= 0) {
                this.state.right -= dir * this.moveSpeed;
            }
        };

        canvas.addEventListener("keyup", handler, true);
        canvas.addEventListener("keydown", handler, true);

        canvas.addEventListener(
            "mousemove",
            (event) => {
                if (this.hasFocus || !this.rotateOnlyIfFocussed) {
                    const currentRotation = this.rotation;
                    currentRotation[1] -= event.movementX * this.rotationSpeed;
                    currentRotation[0] -= event.movementY * this.rotationSpeed;
                    this.rotation = currentRotation;
                }
            },
            false
        );

        document.addEventListener("pointerlockchange", () => {
            this.hasFocus = document.pointerLockElement === canvas;
        });
    }

    updateAndGetViewMatrix(): mat4 {
        const mat = this.getUpRightMatrix();
        const forward: vec3 = [mat[6], mat[7], mat[8]];
        const right: vec3 = [mat[0], mat[1], mat[2]];

        vec3.scale(forward, forward, this.state.forward);
        vec3.scale(right, right, this.state.right);
        vec3.add(right, right, forward);
        vec3.add(this.position, this.position, right);

        return this.getViewMatrix();
    }

    updateAndGetViewProjectionMatrix(): mat4 {
        this.updateAndGetViewMatrix();
        return this.getViewProjectionMatrix();
    }
}
