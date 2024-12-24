namespace SqlSharpLit.Common.ParserLit;

public interface IDatabaseNameProvider
{
    string GetDatabaseNameFromPath(string path);
    void SetDeep(int deep);
}