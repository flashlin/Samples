namespace WebSample.Services;

public class MyGlobalSettings
{
	public string[] Countries { get; set; } = Array.Empty<string>();
	public bool FeatureEnabled { get; set; }
	public List<int> CustomerIds { get; set; } = new();
	public List<BlockedDomain> BlockedDomains { get; set; } = new();
}