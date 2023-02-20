using System.Text.Json;
using T1.WebTools.LocalQueryEx;

namespace QueryApp.Models.Clients;

public class LocalQueryHostClient : ILocalQueryHostClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

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
        return JsonSerializer.Deserialize<T>(responseJson)!;
    }

    private async Task<HttpResponseMessage> PostAsync(string relativeUrl, object request)
    {
        var json = JsonSerializer.Serialize(request);
        var requestContent = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync($"{_baseUrl}{relativeUrl}", requestContent);
        response.EnsureSuccessStatusCode();
        return response;
    }


    private async Task PostJsonVoidAsync(string relativeUrl, object request)
    {
        _ = await PostAsync(relativeUrl, request);
    }
}