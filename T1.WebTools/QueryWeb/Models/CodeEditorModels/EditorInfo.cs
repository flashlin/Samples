using System.Text.Json.Serialization;

namespace QueryWeb.Models.CodeEditorModels;

public class EditorInfo
{
    [JsonPropertyName("prev")]
    public string PrevLine { get; set; } = string.Empty;
    [JsonPropertyName("line")]
    public string Line { get; set; } = string.Empty;
    [JsonPropertyName("after")]
    public string AfterLine { get; set; } = string.Empty;
}