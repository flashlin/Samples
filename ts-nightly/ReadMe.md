
Accessor object Sample Code
```
const user = new User();
const obj = new Accessor(user, x => x.name);
user.name = 'Mary';
const message = obj.getValue();
expect(message).toBe('Mary');
```  

Mock function Sample Code
```
function sayHello(id: number): string {
  return `${id} hello`;
}

function mockSayHello(id: number): string {
  return `mock ${id} hello`;
}

const f = MockFuncCall(true, mockSayHello, sayHello);
const result = f(1);
console.log(result);
```

Mock class sample code
```
class Sample {
    @MockMethod(true, "mock data")
    getData() {
        return "real data"
    }
}
```


```
class Sample {
    @MockMethod(true, {"name": "Mr.Brain", "email": "<EMAIL>"})
    getUser() {
        return {
            "name": "Mr.Jack",
            "email": "<EMAIL>"
        }
    }
}
```

Create WebApi Sample Code
```
const api = new WebApi();
export function getUser() {
    return api.postAsync('url', data);
}
```
