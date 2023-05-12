class MyClass {
  name = "flash";
  sayHello() {
    return "Hello " + this.name;
  }
}

const instance = new MyClass();

const useMy = () => {
  return instance;
}

describe('create object', () => {
    it('case1', () => {
      const instance1 = useMy();
      const { name } = instance1;
      const sayHello = instance1.sayHello.bind(instance1);
      expect(name).toStrictEqual("flash");
      expect(sayHello()).toStrictEqual("Hello flash");
    });
});