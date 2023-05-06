namespace QueryWeb.Models.Clients;

public interface IPredictNextWordsClient
{
    Task<InferResponse> Infer(string text);
    Task AddSql(string sqlCode);
    Task QuerySql();
}