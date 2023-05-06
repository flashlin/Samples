using System.Text.Json.Serialization;

namespace QueryWeb.Models.Clients;

public class InferResponse
{
    [JsonPropertyName("top_k")] 
    public List<InferNextWords> TopK { get; set; } = new();
}