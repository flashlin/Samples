using System.Net.Http.Headers;

namespace T1.SlackSdk;

public class SlackFileDownloader
{
    private readonly string _slackBotToken;

    public SlackFileDownloader(string slackBotToken)
    {
        _slackBotToken = slackBotToken;
    }

    public async Task<bool> DownloadFileAsync(string fileUrl, string localFilePath)
    {
        try
        {
            using var httpClient = new HttpClient();
            httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", _slackBotToken);
            var response = await httpClient.GetAsync(fileUrl);
            response.EnsureSuccessStatusCode();
            await using var fileStream = new FileStream(localFilePath, FileMode.Create, FileAccess.Write, FileShare.None);
            await response.Content.CopyToAsync(fileStream);

            //using var request = new HttpRequestMessage(HttpMethod.Get, fileUrl);
            //request.Headers.Add("Authorization", $"Bearer {_slackBotToken}");
            //httpClient.DefaultRequestHeaders.Add("User-Agent", "SlackFileDownloader/1.0");
            //using var response = await httpClient.SendAsync(request);
            // await using var contentStream = await response.Content.ReadAsStreamAsync();
            // await using var fileStream = new FileStream(localFilePath, FileMode.Create, FileAccess.Write);
            // await contentStream.CopyToAsync(fileStream);
            return true;
        }
        catch (HttpRequestException ex)
        {
            Console.WriteLine(ex.Message);
            return false;
        }
    }
}