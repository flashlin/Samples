using DotNetEnv;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharp;

public class CustomDatabaseNameProvider : IDatabaseNameProvider
{
    private int _deep = 6;

    public void SetDeep(int deep)
    {
        _deep = deep;
    }
    
    public string GetDatabaseNameFromPath(string path)
    {
        return GetNthDirectoryName(path, _deep);
    }
    
    string GetNthDirectoryName(string folder, int n)
    {
        var directoryPath = Path.GetDirectoryName(folder);
        if (directoryPath == null)
        {
            return string.Empty;
        }
        var directories = directoryPath.Split(Path.DirectorySeparatorChar);
        if (n < 0 || n >= directories.Length)
        {
            return string.Empty;
        }
        return directories[n];
    }
}