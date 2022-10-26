using Microsoft.EntityFrameworkCore;

namespace EFSample;

public class Demo
{
    public void TestSqlServer()
    {
        var connectionString = "Data Source=localhost\\SQLExpress;Initial Catalog=North;Integrated Security=true;";
        using var db = new MyDbContext(DbContextOptionsBuilder.UseSqlServer<MyDbContext>(connectionString));
        if (!db.Customers.Any(x => x.Name == "flash"))
        {
            var customer1 = new Customer()
            {
                Name = "flash",
                Birth = DateTime.Parse("2022-11-01"),
                Price = 123
            };
            db.Customers.Add(customer1);
            db.SaveChanges();
        }

        var customerId = db.Customers.Where(x => x.Id == 1)
            .Select(x => x.Id)
            .First();

        var customer2 = new Customer()
        {
            Id = customerId,
            Name = "jack1233",
            Price = 3
        };
        db.Entry(customer2).Property(x => x.Name).IsModified = true;
        db.SaveChanges();
    }
    
    public class TopUser
    {
        public string Name { get; set; } = null!;
    }

    public void TestSqliteMemory()
    {
        using var db = new MyDbContext(DbContextOptionsBuilder.UseSqliteMemory<MyDbContext>("North"));
        db.Database.EnsureCreated();
        //var sql = db.Database.GenerateCreateScript();
        //db.Database.ExecuteSqlRaw(sql);

        // var tables = db.RawSqlQuery($"SELECT name FROM sqlite_schema WHERE type ='table'", dr => new TopUser
        // {
        //     Name = (string)dr[0]
        // });
        
        if (!db.Customers.Any(x => x.Name == "flash"))
        {
            var customer1 = new Customer()
            {
                Name = "flash",
                Birth = DateTime.Parse("2022-11-01"),
                Price = 123
            };
            db.Customers.Add(customer1);
            db.SaveChanges();
        }

        //db.ChangeTracker.Clear();
        var customer2 = db.Customers.AsNoTracking().Where(x => x.Id == 1)
            .Select(x => new Customer
            {
                Id = x.Id,
            })
            .First();

        customer2.Name = "jack";
        db.Entry(customer2).Property(x => x.Name).IsModified = true;
        db.SaveChanges();

        //db.ChangeTracker.Clear();
        var customer3 = db.Customers.First();
    }
}