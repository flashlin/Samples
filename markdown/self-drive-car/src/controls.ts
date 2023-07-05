export class Controls {
    forward = false;
    left = false;
    right = false;
    reverse = false;

    constructor() {
        this.addKeyboardListeners();
    }

    addKeyboardListeners() {
        document.onkeydown = (e) => {
            switch (e.key) {
                case 'ArrowUp':
                    this.forward = true;
                    break;
                case 'ArrowLeft':
                    this.left = true;
                    break;
                case 'ArrowRight':
                    this.right = true;
                    break;
                case 'ArrowDown':
                    this.reverse = true;
                    break;
            }
        };

        document.onkeyup = (e) => {
            switch (e.key) {
                case 'ArrowUp':
                    this.forward = false;
                    break;
                case 'ArrowLeft':
                    this.left = false;
                    break;
                case 'ArrowRight':
                    this.right = false;
                    break;
                case 'ArrowDown':
                    this.reverse = false;
                    break;
            }
        };
    }
}