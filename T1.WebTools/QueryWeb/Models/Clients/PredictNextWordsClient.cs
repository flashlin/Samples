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

    public async Task<InferResponse> InferAsync(string text)
    {
        if (string.IsNullOrEmpty(_config.Url))
        {
            return new InferResponse();
        }
        var resp = await PostJsonAsync<InferResponse>("infer", new
        {
            input = text
        });
        return resp!;
    }
    
    public Task AddSqlAsync(string sqlCode)
    {
        if (string.IsNullOrEmpty(_config.Url))
        {
            return Task.CompletedTask;
        }
        return PostJsonVoidAsync("addsql", new
        {
            sql = sqlCode
        });
    }
    
    public Task QuerySqlAsync()
    {
        return PostJsonAsync<List<string>>("querysql", new {});
    }

    private async Task<T?> PostJsonAsync<T>(string requestUrl, object parameters)
    {
        var message = await PostAsJsonAsync(requestUrl, parameters);
        var responseStream = await message.Content.ReadAsStreamAsync();
        var resp = await JsonSerializer.DeserializeAsync<T>(responseStream, _options);
        return resp;
    }
    
    private Task PostJsonVoidAsync(string requestUri, object parameters)
    {
        return PostAsJsonAsync(requestUri, parameters);
    }
    
    private async Task<HttpResponseMessage> PostAsJsonAsync(string apiUrl, object parameters)
    {
        var message = await _httpClient.PostAsJsonAsync(_config.Url + "/" + apiUrl, parameters, _options);
        message.EnsureSuccessStatusCode();
        return message;
    }
}