
import('http://192.168.2.2:3001/dist/my-lib.js')
.then((module) => {
   var rc = module.sayHello();
   console.log("ts call sayHello", rc);
});

console.log("test");
