namespace SqlBoyLib;

public static class Db
{
    public static SqlQueryBuilder<TEntity> From<TEntity>(string tableName) where TEntity : class
    {
        return new SqlQueryBuilder<TEntity>(tableName);
    }
}

