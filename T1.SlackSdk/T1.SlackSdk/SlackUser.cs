namespace T1.SlackSdk;

public class SlackUser
{
    public static SlackUser Empty => new SlackUser();
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
}