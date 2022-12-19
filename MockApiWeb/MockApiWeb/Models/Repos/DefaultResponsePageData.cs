namespace MockApiWeb.Models.Repos;

public class DefaultResponsePageData
{
    public List<WebApiMockInfoEntity> PageData { get; set; } = new();
    public bool HasNextPage { get; set; }
}