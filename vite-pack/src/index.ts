export class MyClass {
  sayHello() {
    return "Hello";
  }
}

// 在全局範圍內暴露 MyClass
window.MyLib = { MyClass };
