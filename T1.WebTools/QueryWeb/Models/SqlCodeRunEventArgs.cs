using BlazorMonaco;

namespace QueryWeb.Models;

public class SqlCodeRunEventArgs : EventArgs
{
    public string SqlCode { get; set; } = string.Empty;
}

public class KeyEventArgs : EventArgs
{
    public KeyCode KeyCode { get; set; }
    public bool CtrlKey { get; set; }
}