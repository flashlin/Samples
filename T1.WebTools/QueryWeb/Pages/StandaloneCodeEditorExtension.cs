using System.Text;
using BlazorMonaco;
using BlazorMonaco.Editor;

namespace QueryWeb.Pages;

public static class StandaloneCodeEditorExtension
{
    public static async Task<string> GetTextOfBeforeCursorAsync(this StandaloneCodeEditor editor)
    {
        var allText = await editor.GetValue();
        var cursorPosition = await editor.GetPosition()!;
        return GetTextToPosition(allText, cursorPosition);
    }

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