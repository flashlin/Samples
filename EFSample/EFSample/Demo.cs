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
            Name = "jack",
            Price = 3
        };
        db.Entry(customer2).Property(x => x.Name).IsModified = true;
        db.SaveChanges();
    }

    public void TestSqliteMemory()
    {
        SQLitePCL.Batteries.Init();
        using var db = new MyDbContext(DbContextOptionsBuilder.UseSqliteMemory<MyDbContext>("North"));
        var ok = db.Database.EnsureCreated();
        
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
            Name = "jack",
            Price = 3
        };
        db.Entry(customer2).Property(x => x.Name).IsModified = true;
        db.SaveChanges();
    }
}