using LanguageExt.Common;
using T1.Standard.Extensions;
using T1.Standard.IO;

namespace CloneDatabase;

public interface ISqlScriptBuilder
{
    string CreateTableSql(TableInfo tableInfo);
    string DeleteTableSql(string tableName);
}

public class MsSqlScriptBuilder : ISqlScriptBuilder
{
    public string CreateTableSql(TableInfo tableInfo)
    {
        var wr = new IndentStringBuilder();
        wr.WriteLine($"CREATE TABLE [{tableInfo.Name}] (");
        wr.Indent++;
        foreach (var item in tableInfo.Fields.Select((value, idx) => new { idx, value }))
        {
            var field = item.value;
            var nullable = field.IsNullable ? "" : "ISNULL";
            var comma = (item.idx < tableInfo.Fields.Count - 1) ? "," : "";
            wr.WriteLine($"{field.Name} {field.DataType}{GetDeclareDataSize(field)} {nullable} {comma}");
        }
        wr.Indent--;
        wr.WriteLine(")");
        return wr.ToString();
    }

    public string DeleteTableSql(string tableName)
    {
        var wr = new IndentStringBuilder();
        wr.WriteLine($"DROP TABLE [{tableName}]");
        return wr.ToString();
    }


    private string GetDeclareDataSize(TableFieldInfo fieldInfo)
    {
        if( fieldInfo.DataType.IgnoreCaseIndexOf(new []{ "decimal", "float" }) >=0 )
        {
            return $"({fieldInfo.MaxLength},{fieldInfo.Scale})";
        }
        return $"({fieldInfo.MaxLength})";
    }
}

public static class StringCompareExtension
{
    public static bool IgnoreCaseEquals(this string str, string other)
    {
        return str.Equals(other, StringComparison.OrdinalIgnoreCase);
    }

    public static int IgnoreCaseIndexOf(this string str, string[] strList)
    {
        return Array.FindIndex(strList, m => m.IndexOf(str, StringComparison.OrdinalIgnoreCase) >= 0);
    }
}