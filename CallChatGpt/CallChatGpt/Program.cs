// See https://aka.ms/new-console-template for more information

// API 金鑰

string apiKey = File.ReadAllText("D:/Demo/Joe-chatGpt.key");
            
// API 要求 URL
string apiUrl = "https://api.openai.com/v1/engines/davinci-codex/completions";
            
// API 要求內容，包含要求文本和其他參數
string requestData = "{\"prompt\":\"How to use C# to call OpenAI API?\",\"max_tokens\":100,\"temperature\":0.5}";

// 設置 HttpClient 的預設請求標頭
using var client = new HttpClient();
client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
client.DefaultRequestHeaders.Add("Content-Type", "application/json");

// 發送 POST 請求，獲取 OpenAI 的回應
var response = await client.PostAsync(apiUrl, new StringContent(requestData, Encoding.UTF8, "application/json"));

// 如果請求成功，則獲取回應內容並輸出到控制台
if (response.IsSuccessStatusCode)
{
    string responseBody = await response.Content.ReadAsStringAsync();
    Console.WriteLine(responseBody);
}


Console.WriteLine("Hello, World!");
