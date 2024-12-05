namespace SqlSharpLit.Common.ParserLit;

public interface IDatabaseNameProvider
{
    string GetDatabaseNameFromPath(string path);
}