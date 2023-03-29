using QueryKits.CsvEx;
using QueryKits.ExcelUtils;

namespace QueryKits.Services;

public interface IQueryService
{
    List<ExcelSheet> QueryRawSql(string sql);
    void AddSqlCode(string sqlCode);
    List<string> GetTop10SqlCode();
}