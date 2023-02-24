using T1.WebTools.CsvEx;

namespace QueryApp.Models;

public class QueryRawSqlResponse
{
    public string ErrorMessage { get; set; } = string.Empty;
    public CsvSheet CsvSheet { get; set; } = new();
}