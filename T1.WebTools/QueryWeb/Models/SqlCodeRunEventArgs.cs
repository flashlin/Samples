namespace QueryWeb.Models;

public class SqlCodeRunEventArgs : EventArgs
{
    public string SqlCode { get; set; } = string.Empty;
}

public class MultipleSelectedArgs : EventArgs
{
    public List<SelectItem> ItemsSelected { get; set; } = new();
}

public class SelectedArgs : EventArgs
{
    public SelectItem ItemSelected { get; set; } = new();
}