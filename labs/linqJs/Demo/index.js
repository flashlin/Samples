"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
// Import Hello function from linqjs
const linqjs_1 = require("linqjs");
// Demo: use Hello function
function runDemo() {
    const message = (0, linqjs_1.Hello)('World');
    console.log(message);
}
runDemo();
