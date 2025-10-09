namespace SqlBoyLib;

// Example entity class
public class Customer
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public int Age { get; set; }
    public DateTime CreatedAt { get; set; }
}

public static class Examples
{
    public static void BasicExample()
    {
        // Simple query with WHERE clause
        var query = Db.From<Customer>("Customers")
            .Where(x => x.Id > 1 && x.Name == "John")
            .Build();

        string sql = query.ToExecuteSql();
        Console.WriteLine(sql);
    }

    public static void ComplexWhereExample()
    {
        // Complex WHERE conditions
        var query = Db.From<Customer>("Customers")
            .Where(x => (x.Age > 18 && x.Age < 65) || x.Name == "Admin")
            .OrderBy(x => x.Name)
            .Build();

        string sql = query.ToExecuteSql();
        Console.WriteLine(sql);
    }

    public static void OrderByExample()
    {
        // Multiple ORDER BY clauses
        var query = Db.From<Customer>("Customers")
            .Where(x => x.Age > 18)
            .OrderBy(x => x.Name)
            .OrderByDescending(x => x.Age)
            .Build();

        string sql = query.ToExecuteSql();
        Console.WriteLine(sql);
    }

    public static void FullExample()
    {
        // Complete example with all features
        var query = Db.From<Customer>("Customers")
            .Where(x => x.Id > 1 && x.Name == "John")
            .OrderBy(x => x.Age)
            .Build();

        Console.WriteLine("Statement: " + query.Statement);
        Console.WriteLine("Parameters: " + query.ParameterDefinitions);
        Console.WriteLine("\nFull SQL:");
        Console.WriteLine(query.ToExecuteSql());
    }
}

