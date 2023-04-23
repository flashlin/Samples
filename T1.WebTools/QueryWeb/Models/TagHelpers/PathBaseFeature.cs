namespace QueryWeb.Models.TagHelpers;

public class PathBaseFeature : IPathBaseFeature
{
    public string PathBase { get; init; }

    public string GetPath(string url)
    {
        var href = url.Replace("~/", PathBase + "/");
        return href;
    }
}