namespace QueryWeb.Models;

public class SqlCodeRunEventArgs : EventArgs
{
    public string SqlCode { get; set; } = string.Empty;
}