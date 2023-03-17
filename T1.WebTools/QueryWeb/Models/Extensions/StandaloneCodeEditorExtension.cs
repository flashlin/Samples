using System.Text;
using BlazorMonaco;
using BlazorMonaco.Editor;
using Microsoft.JSInterop;

namespace QueryWeb.Models.Extensions;

public static class StandaloneCodeEditorExtension
{
    public static async Task<string> GetTextOfBeforeCursor(this StandaloneCodeEditor editor)
    {
        var allText = await editor.GetValue();
        var cursorPosition = await editor.GetPosition()!;
        return GetTextToPosition(allText, cursorPosition);
    }
    
    // public static async Task Insert(this StandaloneCodeEditor editor, string text)
    // {
    //     var allText = await editor.GetValue();
    //     var before = await editor.GetTextOfBeforeCursor();
    //     var after = allText.Substring(before.Length);
    //     var newAllText = before + text + after;
    //     await editor.SetValue(newAllText);
    // }

    private static string GetTextToPosition(string allText, Position position)
    {
        var sr = new StringReader(allText);
        var lineNumber = 1;
        var text = new StringBuilder();
        do
        {
            var line = sr.ReadLine();
            if (line == null)
            {
                break;
            }
            if (lineNumber == position.LineNumber)
            {
                var s = line.Substring(0, position.Column - 1);
                text.Append(s);
                break;
            }
            text.AppendLine(line);
            lineNumber++;
        } while (true);
        return text.ToString();
    }
}