using System;

namespace ExampleUsage
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("T1.EfCodeFirstGenerator Example");
            Console.WriteLine("================================");
            Console.WriteLine();
            Console.WriteLine("This project demonstrates the usage of T1.EfCodeFirstGenerator.");
            Console.WriteLine();
            Console.WriteLine("Steps:");
            Console.WriteLine("1. Configure your database connection in example.db");
            Console.WriteLine("2. Build the project");
            Console.WriteLine("3. The generator will create:");
            Console.WriteLine("   - {ServerName}_{DatabaseName}.schema file");
            Console.WriteLine("   - DbContext class");
            Console.WriteLine("   - Entity classes");
            Console.WriteLine("   - EntityConfiguration classes");
            Console.WriteLine();
            Console.WriteLine("Generated code will be available in your compilation.");
            Console.WriteLine();
            Console.WriteLine("Example usage:");
            Console.WriteLine("  using var context = new SampleDbDbContext();");
            Console.WriteLine("  var users = context.Users.ToList();");
        }
    }
}

