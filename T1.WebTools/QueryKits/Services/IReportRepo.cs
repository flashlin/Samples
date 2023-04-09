using QueryKits.ExcelUtils;
using QueryKits.Extensions;
using T1.Standard.Data.SqlBuilders;

namespace QueryKits.Services;


//[DefaultReturnInterceptor]
public interface IReportRepo
{
    List<string> GetAllTableNames();
    List<Dictionary<string, object>> QueryRawSql(string sql);
    void ReCreateTable(string tableName, List<ExcelColumn> headers);
    int ImportData(string tableName, ExcelSheet rawDataList);
    List<string> GetTop10SqlCode();
    void AddSqlCode(string sqlCode);
    List<QueryDataSet> QueryDapperMultipleRawSql(string sql);
    List<QueryDataSet> QueryMultipleRawSql(string sql);
    void DeleteTable(string tableName);
    void CreateTableByEntity(Type entityType);
    void MergeTable(MergeTableRequest req);
    ISqlBuilder SqlBuilder { get; }
    TableInfo GetTableInfo(string tableName);
}