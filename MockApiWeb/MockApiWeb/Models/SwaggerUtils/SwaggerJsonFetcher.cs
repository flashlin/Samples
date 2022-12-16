using System.Text.Json;
using MockApiWeb.Controllers.Apis;

namespace MockApiWeb.Models.SwaggerUtils;

public class SwaggerJsonFetcher
{
    private readonly HttpClient _httpClient;

    public SwaggerJsonFetcher(IHttpClientFactory httpClientFactory)
    {
        _httpClient = httpClientFactory.CreateClient();
        //httpClient.BaseAddress = new Uri("https://api.line.me");
        //httpClient.DefaultRequestHeaders.Add("authorization", "Bearer {CannelAccessToken}");
    }

    public async Task<SwaggerJsonDocument> Read(string swaggerJsonUrl)
    {
        var url = "https://localhost:44325/swagger/v1/swagger.json";
        var response = await _httpClient.GetAsync(swaggerJsonUrl);
        var content = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<SwaggerJsonDocument>(content) ?? new SwaggerJsonDocument();
    }
}