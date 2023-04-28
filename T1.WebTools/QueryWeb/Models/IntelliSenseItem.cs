namespace QueryWeb.Models;

public enum IntelliSenseItemType
{
    Keyword = 17
}

public class IntelliSenseItem
{
    public string Label { get; set; } = string.Empty;
    public IntelliSenseItemType Kind { get; set; }
    public string Detail { get; set; } = "Keyword";
    public string InsertText { get; set; } = string.Empty;
}