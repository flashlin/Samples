// See https://aka.ms/new-console-template for more information

using System.Text;
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

await using var logFile = new FileStream("d:/demo/1.txt", FileMode.Create);
await using var writer = new StreamWriter(logFile, Encoding.UTF8); 
var response = await client.GetHistoryAsync(supportChannelId, dateRange);
foreach (var item in response)
{
    var message = $"[{item.Time.ToDisplayString()}] {item.User.Name}: {item.Text}";
    Console.WriteLine(message);
    writer.WriteLine(message);
    foreach (var threadMessage in item.ThreadMessages)
    {
        var subMessage = $"   +[{threadMessage.Time.ToDisplayString()}] {threadMessage.User.Name}: {threadMessage.Text}";
        Console.WriteLine(subMessage);
        writer.WriteLine(subMessage);
    }
}
writer.Flush();