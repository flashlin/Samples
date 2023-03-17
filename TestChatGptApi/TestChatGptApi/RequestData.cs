using System.Text.Json.Serialization;

namespace TestChatGptApi;

public class RequestData
{
    public string Prompt { get; set; }
    
    [JsonPropertyName("max_tokens")] 
    public int MaxTokens {get;set;}
    public float Temperature { get; set; }
}