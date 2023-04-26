// See https://aka.ms/new-console-template for more information


using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using TestChatGptApi;

var apiKey = File.ReadAllText("d:/demo/joe-chat-gpt.key");
//var apiUrl = "https://api.openai.com/v1/engines/davinci-codex/completions";
            
// // API 要求內容，包含要求文本和其他參數
// var model = "text-davinci-002";
// model = "gpt-3.5-turbo";
//
// var req = new RequestData
// {
//     Prompt = "台灣現在的總統是誰?",
//     MaxTokens = 100,
//     Temperature = 0.3f
// };
// var options = new JsonSerializerOptions
// {
//     PropertyNamingPolicy = JsonNamingPolicy.CamelCase
// };
// var requestData = JsonSerializer.Serialize(req, options);
//
// var apiUrl = $"https://api.openai.com/v1/models/{model}/completions";
//
//
// using var client = new HttpClient();
//
// var request = new HttpRequestMessage();
// request.RequestUri = new Uri(apiUrl);
// //request.Headers.Add("Authorization", $"Bearer {apiKey}");
// request.Headers.Add("Ocp-Apim-Subscription-Key", apiKey);
// request.Content = new StringContent(requestData, Encoding.UTF8, "application/json");
//
// Console.WriteLine($"request... key='{apiKey}'");
// var response = await client.SendAsync(request);
//
// Console.WriteLine($"response...{response.StatusCode.ToString()}");
// if (response.IsSuccessStatusCode)
// {
//     var responseBody = await response.Content.ReadAsStringAsync();
//     Console.WriteLine(responseBody);
// }
var api = new OpenAiProxy(apiKey);
var resp = await api.Send("台灣的總統是誰?");
Console.WriteLine($"{resp}");
Console.ReadKey();