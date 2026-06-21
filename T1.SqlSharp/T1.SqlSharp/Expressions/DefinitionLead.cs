namespace T1.SqlSharp.Expressions;

public static class DefinitionLead
{
    public static string ToSql(bool isAlter, bool isOrAlter, string objectKeyword)
    {
        if (isAlter)
        {
            return $"ALTER {objectKeyword} ";
        }

        return isOrAlter ? $"CREATE OR ALTER {objectKeyword} " : $"CREATE {objectKeyword} ";
    }
}
