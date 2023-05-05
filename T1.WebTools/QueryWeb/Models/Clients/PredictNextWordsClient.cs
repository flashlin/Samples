using System.Text.Json;
using System.Text.Json.Serialization;

namespace QueryWeb.Models.Clients;

public interface IPredictNextWordsClient
{
    Task<string> Infer(string text);
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

    public async Task<string> Infer(string text)
    {
        return await PostJsonAsync<string>("infer", new
        {
            input = text
        });
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

    private async Task<T> PostJsonAsync<T>(string requestUri, object parameters)
    {
        var message = await PostJsonAsync(requestUri, parameters);
        var responseStream = await message.Content.ReadAsStreamAsync();
        var resp = await JsonSerializer.DeserializeAsync<T>(responseStream, _options);
        return resp!;
    }
    
    private Task PostJsonVoidAsync(string requestUri, object parameters)
    {
        return PostJsonAsync(requestUri, parameters);
    }
    
    private async Task<HttpResponseMessage> PostJsonAsync(string requestUri, object parameters)
    {
        var message = await _httpClient.PostAsJsonAsync("http://127.0.0.1:5001/" + requestUri, parameters, _options);
        message.EnsureSuccessStatusCode();
        return message;
    }
}