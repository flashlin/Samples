The purpose of this library is to extend or support the functionality of Entity Framework.


Here are examples for Upsert a single record:
```csharp
var entity = new Entity
{
    Id = 1,
    Name = "Joho"
};

_db.Upsert(entity)
    .On(x => x.Id)
    .Execute();
```

Here are examples for Upsert multiple records:
```csharp
var entityArray = [
    new Entity
    {
        Id = 1,
        Name = "John"
    },
    new Entity
    {
        Id = 2,
        Name = "Jack"
    },
];

_db.Upsert(entityArray)
    .On(x => x.Id)
    .Execute();
```

Here are examples for Upsert a single record with multiple keys:
```csharp
var entityArray = [
    new Entity
    {
        Id = 1,
        Name = "John"
    },
];

_db.Upsert(entityArray)
    .On(x => new {x.Id, x.Name})
    .Execute();
```

Here are examples for Upsert large records:
```csharp
_db.UpsertRange(entityArray)
    .On(x => x.Id)
    .Execute();
```

