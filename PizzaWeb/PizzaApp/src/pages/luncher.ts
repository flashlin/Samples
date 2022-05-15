import { MyHelper } from "../models/helper";

function sayHello() {
	let h = new MyHelper();
	console.log('hello' + h.getName());
}

sayHello();