using System.Text.Json;
using Microsoft.Extensions.Options;

namespace QueryWeb.Models.Clients;

public class PredictNextWordsClient : IPredictNextWordsClient
{
    private readonly HttpClient _httpClient;

    private readonly JsonSerializerOptions _options = new JsonSerializerOptions
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        ReadCommentHandling = JsonCommentHandling.Disallow,
    };

    private readonly PredictNextWordsConfig _config;

    public PredictNextWordsClient(IOptions<PredictNextWordsConfig> config, IHttpClientFactory httpClientFactory)
    {
        _httpClient = httpClientFactory.CreateClient();
        _config = config.Value;
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
        var message = await _httpClient.PostAsJsonAsync(_config.Url + "/" + requestUri, parameters, _options);
        message.EnsureSuccessStatusCode();
        return message;
    }
}