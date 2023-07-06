import { SampleClass } from './sample';
import { MockFuncCall } from '../src/mock';

function sayHelloNoArgs(): string {
  return `Hello`;
}

function mockSayHelloNoArgs(): string {
  return `Mock`;
}

function sayHello(id: number): string {
  return `${id} hello`;
}

function mockSayHello(id: number): string {
  return `mock ${id} hello`;
}


function sayHelloAsync(id: number): Promise<string> {
  return Promise.resolve(`${id} helloAsync`);
}


function mockSayHelloAsync(id: number): Promise<string> {
  return Promise.resolve(`mock ${id} helloAsync`);
}


test('mock func no args', () => {
  const f = MockFuncCall(true, mockSayHelloNoArgs, sayHelloNoArgs);
  const message = f();
  expect(message).toBe('Mock');
});

test('mock func', () => {
  const f = MockFuncCall(true, mockSayHello, sayHello);
  const message = f(123);
  expect(message).toBe('mock 123 hello');
});


test('mock func false', () => {
  const f = MockFuncCall(false, mockSayHello, sayHello);
  const message = f(123);
  expect(message).toBe('123 hello');
});

test('mock funcAsync', async () => {
  const f = MockFuncCall(true, mockSayHelloAsync, sayHelloAsync);
  const message = await f(123);
  expect(message).toBe('mock 123 helloAsync');
});


test('mock class member', async () => {
  const a = new SampleClass();
  const message = a.sayHello('Flash');
  expect(message).toBe('Flash');
});

test('mock class member with hard code', async () => {
  const a = new SampleClass();
  const message = a.sayHello2('Flash');
  expect(message).toBe('XXX');
});

test('mock class member with this', async () => {
  const a = new SampleClass();
  const message = a.sayHello3('Flash');
  expect(message).toBe('Flash 123');
});

test('mock class member return null', async () => {
  const a = new SampleClass();
  const message = a.sayHello4('Flash');
  expect(message).toBe(null);
});