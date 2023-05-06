using System.Text.Json.Serialization;

namespace QueryWeb.Models.Clients;

public class InferResponse
{
    [JsonPropertyName("next_words")] 
    public List<InferNextWords> top_k { get; set; } = new();
}