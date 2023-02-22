namespace QueryApp.Models;

public class QueryRawSqlResponse
{
    public List<Dictionary<string, string>> Data { get; set; } = new();
    public string ErrorMessage { get; set; } = string.Empty;
}