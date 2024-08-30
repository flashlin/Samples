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
var oldest = new DateTime(2024, 8, 27).ToUnixTimeSeconds();
var latest = new DateTime(2024, 8, 28).ToUnixTimeSeconds();


var response = await client.Conversations.History(
    supportChannelId,
    latest.ToString(),
    oldest.ToString(),
    inclusive: false,
    limit: 100
);

foreach (var message in response.Messages)
{
    Console.WriteLine($"User: {message.User}, Text: {message.Text}, Timestamp: {message.Ts}");
}
Console.ReadKey();