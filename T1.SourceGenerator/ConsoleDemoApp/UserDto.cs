namespace ConsoleDemoApp;

public class UserDto
{
    public string Name { get; set; } = string.Empty;
    public float Level { get; set; }
    public float Price { get; }
    public DateTime Birth { get; set; }
}