using System.Text.Json;
using System.Text.Json.Serialization;

namespace QueryWeb.Models.Clients;

public interface IPredictNextWordsClient
{
    Task<InferResponse> Infer(string text);
    Task AddSql(string sqlCode);
    Task QuerySql();
}

public class PredictNextWordsClient : IPredictNextWordsClient
{
    private readonly HttpClient _httpClient;

    private readonly JsonSerializerOptions _options = new JsonSerializerOptions
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        ReadCommentHandling = JsonCommentHandling.Disallow,
    };

    public PredictNextWordsClient(IHttpClientFactory httpClientFactory)
    {
        _httpClient = httpClientFactory.CreateClient();
    }

    public async Task<InferResponse> Infer(string text)
    {
        var resp = await PostJsonAsync<InferResponse>("infer", new
        {
            input = text
        });
        return resp!;
    }
    
    public Task AddSql(string sqlCode)
    {
        return PostJsonVoidAsync("addsql", new
        {
            sql = sqlCode
        });
    }
    
    public Task QuerySql()
    {
        return PostJsonVoidAsync("querysql", new {});
    }

    private async Task<T?> PostJsonAsync<T>(string requestUri, object parameters)
    {
        var message = await PostJsonAsync(requestUri, parameters);
        var responseStream = await message.Content.ReadAsStreamAsync();
        var resp = await JsonSerializer.DeserializeAsync<T>(responseStream, _options);
        return resp;
    }
    
    private Task PostJsonVoidAsync(string requestUri, object parameters)
    {
        return PostJsonAsync(requestUri, parameters);
    }
    
    private async Task<HttpResponseMessage> PostJsonAsync(string requestUri, object parameters)
    {
        var message = await _httpClient.PostAsJsonAsync("http://127.0.0.1:8000/" + requestUri, parameters, _options);
        message.EnsureSuccessStatusCode();
        return message;
    }
}

public class InferNextWords
{
    public string next_words { get; set; }
    public float probability { get; set; }
}

public class InferResponse
{
    public List<InferNextWords> top_k { get; set; } = new();
}