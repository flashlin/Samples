using Microsoft.EntityFrameworkCore;

namespace ConsoleDemoApp;

public class UserDto
{
    public string Name { get; set; } = string.Empty;
    public float Level { get; set; }
    public float Price { get; }
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
   public DbSet<UserEntity> Users { get; set; }
}