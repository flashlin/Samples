namespace QueryWeb.Models.Clients;

public interface IPredictNextWordsClient
{
    Task<InferResponse> InferAsync(string text);
    Task AddSqlAsync(string sqlCode);
    Task QuerySqlAsync();
}