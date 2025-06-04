using SqlSharpLit.Common.ParserLit;
using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;

public class GenerateTrainDataService : IGenerateTrainDataService
{
    public void Run()
    {
        foreach (var sqlExpression in EnumerateDataTableSchemas("D:\\coredev_tw\\DbProjects"))
        {
            Console.WriteLine(sqlExpression);
        }
    }
    
    public IEnumerable<ISqlExpression> EnumerateDataTableSchemas(string sqlFolder)
    {
        foreach (var sqlFile in EnumerateSqlFiles(sqlFolder))
        {
            new ExtractSqlHelper();
            var sqlResult = new SqlParser(sqlFile).Parse();
            if (sqlResult.HasError)
            {
                continue;
            }
            yield return sqlResult.ResultValue;
        }
    }

    private static IEnumerable<string> EnumerateSqlFiles(string sqlFolder)
    {
        foreach (var sqlFile in Directory.EnumerateFiles(sqlFolder, "*.sql", SearchOption.AllDirectories))
        {
            yield return sqlFile;
        }
    }
}