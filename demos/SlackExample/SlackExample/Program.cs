// See https://aka.ms/new-console-template for more information

using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Options;
using SlackExample;
using SlackNet;
using SlackNet.WebApi;
using T1.Standard.Extensions;

Console.WriteLine("Hello, World!");
var file = System.IO.File.ReadAllText("d:/demo/configs/slack-robot.json");
var config = JsonSerializer.Deserialize<SlackConfig>(file, new JsonSerializerOptions
{
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
})!;
var client = new SlackClient(Options.Create(config));

var supportChannelId = "CGSP5TB6E";
var today = DateTime.Now;
var yesterday = today.AddDays(-1).Date;
var twoDaysAgo = yesterday.AddDays(-2).Date;
var dateRange = new DateTimeRange
{
    Start = twoDaysAgo,
    End = yesterday
};

var response = await client.GetHistoryAsync(supportChannelId, dateRange).ToListAsync();
foreach (var item in response)
{
    Console.WriteLine($"  {item.Time.ToDisplayString()} {item.User.Name}: {item.Text}");
    foreach (var threadMessage in item.ThreadMessages)
    {
        Console.WriteLine($"  {threadMessage.Time.ToDisplayString()} {threadMessage.User.Name}: {threadMessage.Text}");
    }
}