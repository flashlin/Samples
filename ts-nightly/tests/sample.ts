import { MockMethod } from "../src/mock";

function mockSayHelloMember(name: string): string {
  return `${name}`;
}

function mockSayHello3Member(this: any, name: string): string {
  return `${name} ${this.id}`;
}

export class SampleClass {
  id = 123;

  @MockMethod(true, mockSayHelloMember)
  sayHello(name: string): string {
    return `Hello ${name}`;
  }

  @MockMethod(true, "XXX")
  sayHello2(name: string): string {
    return `Hello ${name}`;
  }

  @MockMethod(true, mockSayHello3Member)
  sayHello3(name: string): string {
    return `Hello ${name}`;
  }

  @MockMethod(true, null)
  sayHello4(name: string): string {
    return `Hello ${name}`;
  }
}