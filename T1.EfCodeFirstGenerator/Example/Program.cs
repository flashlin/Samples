using System;
using Generated;

namespace ExampleUsage
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("T1.EfCodeFirstGenerateCli Example");
            Console.WriteLine("==================================");
            Console.WriteLine();
            Console.WriteLine("This project demonstrates the usage of T1.EfCodeFirstGenerateCli.");
            Console.WriteLine();
            
            // Example: Using the generated DbContext
            Console.WriteLine("Generated DbContext available: SampleDbDbContext");
            Console.WriteLine("Generated Entities: UsersEntity, ProductsEntity");
            Console.WriteLine();
            
            Console.WriteLine("To use the generated code:");
            Console.WriteLine("1. Configure database connection in example.db");
            Console.WriteLine("2. Run: dotnet run --project ../T1.EfCodeFirstGenerateCli -- .");
            Console.WriteLine("3. Build this project to compile the generated code");
            Console.WriteLine();
            
            Console.WriteLine("Example usage:");
            Console.WriteLine("  using var context = new SampleDbDbContext();");
            Console.WriteLine("  context.Database.EnsureCreated();");
            Console.WriteLine("  var users = context.Users.ToList();");
            Console.WriteLine();
            
            Console.WriteLine("Generated files are in: Generated/");
            Console.WriteLine("- SampleDbDbContext.cs");
            Console.WriteLine("- Entities/UsersEntity.cs");
            Console.WriteLine("- Entities/ProductsEntity.cs");
            Console.WriteLine("- Configurations/UsersEntityConfiguration.cs");
            Console.WriteLine("- Configurations/ProductsEntityConfiguration.cs");
        }
    }
}

