import { MockMethod } from "./mock";

function mockSayHelloMember(name: string): string {
  return `${name}`;
}

export class SampleClass {

  @MockMethod(true, mockSayHelloMember)
  sayHello(name: string): string {
    return `Hello ${name}`;
  }
}