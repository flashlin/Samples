using GitCli.Models.ConsoleMixedReality;

namespace GitCliTest;

[TestFixture]
public class TextAreaTest
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void input_two_lines_and_select_cross_lines_and_delete()
    {
        var textArea = new TextArea(new Rect
        {
            Left = 0,
            Top = 0,
            Width = 3,
            Height = 2
        });
        
        textArea.Keyin("12345");
        textArea.ShiftKey(ConsoleKey.LeftArrow, false);
        textArea.ShiftKey(ConsoleKey.LeftArrow);
        textArea.ShiftKey(ConsoleKey.LeftArrow);
        textArea.ShiftKey(ConsoleKey.Delete, false);

        Assert.That(textArea.Value, Is.EqualTo("125"));

        Assert.That(textArea.CursorPosition, Is.EqualTo(new Position
        {
            X = 2,
            Y = 0,
        }));
    }
}