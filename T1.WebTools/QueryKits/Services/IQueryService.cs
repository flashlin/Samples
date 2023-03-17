using QueryKits.CsvEx;

namespace QueryKits.Services;

public interface IQueryService
{
    List<CsvSheet> QueryRawSql(string sql);
}