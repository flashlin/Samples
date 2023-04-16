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

public class MultipleSelectedArgs : EventArgs
{
    public List<SelectItem> SelectedItems { get; set; } = new();
}