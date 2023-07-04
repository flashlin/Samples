import { MockFuncCall, MockMethod } from '../src/mock';

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