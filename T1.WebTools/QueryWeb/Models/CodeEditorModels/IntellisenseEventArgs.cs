namespace QueryWeb.Models.CodeEditorModels;

public class IntellisenseEventArgs : EventArgs
{
    public EditorInfo EditorInfo { get; set; } = new();

    public List<IntelliSenseItem> Suggestions { get; set; } = new();
}