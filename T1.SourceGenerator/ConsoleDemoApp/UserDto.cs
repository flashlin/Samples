using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Query.Internal;
using T1.SourceGenerator.Attributes;

namespace ConsoleDemoApp;

public class UserDto
{
    public string Name { get; set; } = string.Empty;
    public float Level { get; set; }
    public float Price { get; set; }
    public DateTime Birth { get; set; }

    public IQueryable<UserEntity> Test(MyDbContext db, string name)
    {
        return from tb1 in db.Users
            where tb1.Name == name
            select tb1;
    }
}

public class MyDbContext : DbContext
{
    public DbSet<UserEntity> Users { get; set; } = null!;

    [LinqExpressionCompile]
    private static readonly Func<MyDbContext, int, IQueryable<UserEntity>> GetUserByIdInternal =
        (MyDbContext context, int id) =>
            from e in context.Users
            where e.Id == id
            select e;

    public UserEntity GetUser(int id)
    {
        var q1 = GetUserByIdInternal(this, id);
        return q1.First();
    }
}