using System.Text.Json;
using QueryKits.Services;
using T1.WebTools.LocalQueryEx;

namespace QueryApp.Models.Clients;

public class LocalQueryHostClient : ILocalQueryHostClient
{
    private readonly string _baseUrl;
    private readonly HttpClient _httpClient;
    private static readonly JsonSerializerOptions JsonOptions = CreateDefaultSerializeOptions();

    public LocalQueryHostClient(IHttpClientFactory httpClientFactory)
    {
        _httpClient = httpClientFactory.CreateClient();
        _baseUrl = "http://127.0.0.1:5004/api/LocalQueryHost/";
    }

    public async Task<EchoResponse> EchoAsync(ILocalEnvironment localEnvironment)
    {
        var resp = await PostJsonAsync<EchoResponse>("echo", new EchoRequest
        {
            AppUid = localEnvironment.AppUid,
            Port = localEnvironment.Port,
        });
        return resp;
    }

    private async Task<T> PostJsonAsync<T>(string relativeUrl, object request)
    {
        var response = await PostAsync(relativeUrl, request);
        var responseJson = await response.Content.ReadAsStringAsync();
        return DeserializeObject<T>(responseJson);
    }

    private static T DeserializeObject<T>(string json)
    {
        return JsonSerializer.Deserialize<T>(json, JsonOptions)!;
    }

    private static JsonSerializerOptions CreateDefaultSerializeOptions()
    {
        var option = new JsonSerializerOptions()
        {
            PropertyNameCaseInsensitive = true
        };
        return option;
    }

    private async Task<HttpResponseMessage> PostAsync(string relativeUrl, object request)
    {
        var json = SerializeObject(request);
        var requestContent = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync($"{_baseUrl}{relativeUrl}", requestContent);
        response.EnsureSuccessStatusCode();
        return response;
    }

    private static string SerializeObject(object instance)
    {
        var json = JsonSerializer.Serialize(instance, JsonOptions);
        return json;
    }

    private async Task PostJsonVoidAsync(string relativeUrl, object request)
    {
        _ = await PostAsync(relativeUrl, request);
    }
}