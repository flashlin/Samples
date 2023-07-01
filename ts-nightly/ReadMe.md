
```
@MockMethod(true, {"name": "Mr.Brain", "email": "<EMAIL>"})
export function getUser() {
    return {
        "name": "Mr.Jack",
        "email": "<EMAIL>"
    }
}
```

```
const api = new WebApi();
export function getUser() {
    return api.postAsync('url', data);
}
```
