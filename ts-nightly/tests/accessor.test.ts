import { SampleClass } from './sample';
import { Accessor } from '../src/common';

class User {
  name = "Jack";
}


test('getValue', () => {
  const user = new User();
  const obj = new Accessor(user, x => x.name);
  user.name = 'Mary';
  const message = obj.getValue();
  expect(message).toBe('Mary');
});


test('setValue', () => {
  const user = new User();
  const obj = new Accessor(user, x => x.name);
  obj.setValue('Flash')
  const message = user.name;
  expect(message).toBe('Flash');
});
