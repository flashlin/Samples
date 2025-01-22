using T1.SqlSharp.Expressions;
using T1.SqlSharp.ParserLit;

namespace SqlSharpLit.Common.ParserLit;

public class ExtractSqlFileHelper
{
    public IEnumerable<string> GetSqlFiles(string folder)
    {
        if (!Directory.Exists(folder))
        {
            yield break;
        }

        var files = Directory.GetFiles(folder, "*.sql");
        foreach (var file in files)
        {
            yield return file;
        }

        var subFolders = Directory.GetDirectories(folder);
        foreach (var subFolder in subFolders)
        {
            foreach (var file in GetSqlFiles(subFolder))
            {
                yield return file;
            }
        }
    }

    public IEnumerable<SqlFileContent> GetSqlTextFromFolder(string folder)
    {
        foreach (var sqlFile in GetSqlFiles(folder))
        {
            var sql = File.ReadAllText(sqlFile);
            yield return new SqlFileContent
            {
                FileName = sqlFile,
                Sql = sql,
            };
        }
    }

    public IEnumerable<SqlFileContent> GetSqlContentsFromFolder(string folder)
    {
        foreach (var sqlFile in GetSqlFiles(folder))
        {
            Console.WriteLine($"Parsing {sqlFile}");
            var sql = File.ReadAllText(sqlFile);
            List<ISqlExpression> sqlExpressions;
            try
            {
                sqlExpressions = new SqlParser(sql).Extract().ToList();
            }
            catch (Exception e)
            {
                Console.WriteLine($"Error parsing {sqlFile}");
                Console.WriteLine(e.Message);
                continue;
            }

            yield return new SqlFileContent
            {
                FileName = sqlFile,
                Sql = sql,
                SqlExpressions = sqlExpressions
            };
        }
    }
}