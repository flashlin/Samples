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

```
const api = new WebApi();
export function getUser() {
    return api.postAsync('url', data);
}
```
