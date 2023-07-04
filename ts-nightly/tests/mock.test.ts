
function sayHello(id: number): string {
  return `${id} hello`;
}


test('mock func', () => {
  const message = sayHello(123);
  expect(message).toBe('123 hello');
});
