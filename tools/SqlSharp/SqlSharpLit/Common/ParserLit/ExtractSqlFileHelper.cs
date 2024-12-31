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
}