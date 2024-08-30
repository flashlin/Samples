// See https://aka.ms/new-console-template for more information

using System.Text.Json;
using System.Text.Json.Serialization;
using SlackNet;
using SlackNet.WebApi;

Console.WriteLine("Hello, World!");
var file = System.IO.File.ReadAllText("d:/demo/configs/slack-robot.json");
var config = JsonSerializer.Deserialize<SlackConfig>(file, new JsonSerializerOptions
{
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
})!;
var client = new SlackApiClient(config.Token);

var supportChannelId = "CGSP5TB6E";
var today = DateTime.Now;
var yesterday = today.AddDays(-1).Date;
var twoDaysAgo = yesterday.AddDays(-2).Date;
var oldest = twoDaysAgo.ToUnixTimeSeconds();
var latest = yesterday.ToUnixTimeSeconds();


var response = await client.Conversations.History(
    supportChannelId,
    latest.ToString(),
    oldest.ToString(),
    inclusive: true,
    limit: 100,
    includeAllMetadata: true
);

foreach (var message in response.Messages)
{
    var userName = message.User;
    if (string.IsNullOrEmpty(message.User))
    {
        var userInfo = await client.Users.Info(message.User);
        userName = userInfo.Profile.DisplayName ?? userInfo.Profile.RealName;
    }
    Console.WriteLine($"User: {userName}, Text: {message.Text}, Timestamp: {message.Ts}");
}