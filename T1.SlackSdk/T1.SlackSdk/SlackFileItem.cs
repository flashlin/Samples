namespace T1.SlackSdk;

public class SlackFileItem
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string UrlPrivate { get; set; } = string.Empty;
    public bool IsPublic { get; set; }
}