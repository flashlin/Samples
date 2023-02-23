using QueryApp.Models.Helpers;

namespace QueryApp.Models.Services;

public interface IReportRepo
{
    List<string> GetAllTableNames();
    List<Dictionary<string, object>> QueryRawSql(string sql);
    void ReCreateTable(string tableName, List<ExcelColumn> headers);
    int ImportData(string tableName, ExcelSheet rawDataList);
}