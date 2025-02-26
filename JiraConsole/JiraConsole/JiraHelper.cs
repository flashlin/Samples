using System.Net.Http.Headers;
using System.Net.Http.Json;
using Microsoft.Extensions.Options;

namespace JiraConsole;

public class JiraConfig
{
    public string BaseUrl { get; set; } = string.Empty;
    public string UserName { get; set; } = string.Empty;
    public string ApiKey { get; set; } = string.Empty;
}

public class JiraHelper
{
    private readonly HttpClient _client;

    public JiraHelper(IHttpClientFactory httpClientFactory, IOptions<JiraConfig> jiraConfig)
    {
        _client = httpClientFactory.CreateClient("JiraClient");
        var config = jiraConfig.Value;
        var authData = System.Text.Encoding.UTF8.GetBytes($"{config.UserName}:{config.ApiKey}");
        var basicAuthentication = Convert.ToBase64String(authData);
        _client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", basicAuthentication);
        _client.BaseAddress = new Uri(config.BaseUrl);
    }

    public async Task Get()
    {
        var wikiUrl = "https://ironman.atlassian.net/wiki/spaces/KM/overview?homepageId=56607";

        var response = await _client.GetAsync(wikiUrl);
        var content = await response.Content.ReadAsStringAsync();
        Console.WriteLine(content);
    }
    
    public async Task<JiraTicketResponse> CreateTaskTicket(string projectKey, string summary, string description)
    {
        var data = new
        {
            fields = new
            {
                issuetype = new { id = 10002 /*Task*/ },
                summary = summary,
                project = new { key = projectKey /*Key of your project*/},
                description = new
                {
                    version = 1,
                    type = "doc",
                    content = new[] {
                        new {
                            type = "paragraph",
                            content = new []{
                                new {
                                    type = "text",
                                    text =  description
                                }
                            }
                        }
                    }
                }
            }
        };
        var result = await _client.PostAsJsonAsync("/rest/api/3/issue", data);

        if (result.StatusCode == System.Net.HttpStatusCode.Created)
        {
            return await result.Content.ReadFromJsonAsync<JiraTicketResponse>() ?? throw new InvalidOperationException();
        }
        throw new Exception(await result.Content.ReadAsStringAsync());
    }
}
public class JiraTicketResponse
{
    public int Id { get; set; }
    public string Key { get; set; }
    public Uri Self { get; set; }
}