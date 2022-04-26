using CsvCli.Repositories;

namespace CsvCli.Services;

public class ConsoleApp
{
    private LocalDbContext _db;

    public ConsoleApp(LocalDbContext db)
    {
        _db = db;
    }
    
    public void Run(string[] args)
    {
        Console.WriteLine($"Csv Command Interface {args.Length}");
        var commandName = args[0];
        if (commandName == "import")
        {
            var csvFile = args[1];
            var tableName = args[2];
            Console.WriteLine($"{csvFile} {tableName}");
            _db.ImportCsvFile(csvFile, tableName);
        }
    }
}