using BlazorMonaco;

namespace QueryWeb.Models.CodeEditorModels;

public class KeyEventArgs : EventArgs
{
    public KeyCode KeyCode { get; set; }
    public bool CtrlKey { get; set; }
}