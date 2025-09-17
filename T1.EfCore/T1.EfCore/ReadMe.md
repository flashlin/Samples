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


```csharp
public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options)
    : base(options)
    {
    }

    public DbSet<Blog> Blogs => Set<Blog>();
    public DbSet<Post> Posts => Set<Post>();

    protected override void OnConfiguring(DbContextOptionsBuilder
        optionsBuilder)
    {
        optionsBuilder.AddNPlusOneDetector(new NPlusOneDetectorOptions()
        {
            CaptureStackTrace = true,
            LogToConsole = true,
            Threshold = 5,
            DetectionWindowMs = 2000,
            CooldownMs = 3000,
            CleanupIntervalMinutes = 5,
            OnDetection = (result) =>
            {
                Console.WriteLine($"N+1 DETECTED: {result.ExecutionCount} queries in {result.DurationMs:F2}ms");
                Console.WriteLine($"Location: {result.StackTrace?.Split('\n').FirstOrDefault()?.Trim()}");
                Console.WriteLine($"Query: {(result.Query.Length > 80 ? result.Query.Substring(0, 80) + "..." : result.Query)}");
                Console.WriteLine($"Time: {result.DetectedAt:HH:mm:ss}");
                Console.WriteLine($"Context: {result.DbContextType}");
                Console.WriteLine(new string('-', 50));
            }
        });
    }
}
```
