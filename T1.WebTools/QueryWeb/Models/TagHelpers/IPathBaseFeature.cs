namespace QueryWeb.Models.TagHelpers;

public interface IPathBaseFeature
{
    public string PathBase { get; init; }
    string GetPath(string url);
}