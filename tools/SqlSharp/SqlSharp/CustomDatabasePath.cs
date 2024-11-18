using DotNetEnv;
using SqlSharpLit.Common.ParserLit;

namespace SqlSharp;

public class CustomDatabaseNameProvider : IDatabaseNameProvider
{
    public string GetDatabaseNameFromPath(string path)
    {
        return GetNthDirectoryName(path, 3);
        Env.Load();
        var databaseFolders = Env.GetString("DATABASE_FOLDERS").Split("\n");
        foreach(var dbPath in databaseFolders)
        {
            var idx = path.IndexOf(dbPath, StringComparison.Ordinal);
            if(idx==-1){
                continue;
            }
            var startIdx = idx + dbPath.Length;
            var endIdx = path.Substring(startIdx).IndexOf("\\", StringComparison.Ordinal);
            if(endIdx == -1){
                return path.Substring(startIdx);
            }
            return path.Substring(startIdx, endIdx);
        }
        return string.Empty;
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