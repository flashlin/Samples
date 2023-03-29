using QueryKits.ExcelUtils;
using QueryKits.Extensions;

namespace QueryKits.Services;


[DefaultReturnInterceptor]
public interface IReportRepo
{
    List<string> GetAllTableNames();
    List<Dictionary<string, object>> QueryRawSql(string sql);
    void ReCreateTable(string tableName, List<ExcelColumn> headers);
    int ImportData(string tableName, ExcelSheet rawDataList);
    List<string> GetTop10SqlCode();
    void AddSqlCode(string sqlCode);
    IEnumerable<QueryDataSet> QueryMultipleRawSql(string sql);
}