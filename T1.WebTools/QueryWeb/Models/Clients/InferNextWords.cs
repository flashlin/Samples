using System.Text.Json.Serialization;

namespace QueryWeb.Models.Clients;

public class InferNextWords
{
    [JsonPropertyName("next_words")] 
    public string NextWords { get; set; } = string.Empty;
    [JsonPropertyName("probability")]
    public float Probability { get; set; }
}