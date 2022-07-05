using GitCli.Models.ConsoleMixedReality;

namespace GitCliTest;

public class ConsoleTextBoxTest
{
    private TextBox _textbox;

    [SetUp]
    public void Setup()
    {
        _textbox = new TextBox(new Rect
        {
            Left = 10,
            Top = 10,
            Width = 10,
            Height = 1
        });
    }

    [Test]
    public void InputAbc()
    {
        _textbox.Keyin("abc");
        Assert.That(_textbox.EditIndex, Is.EqualTo(3));
    }
  
    [Test]
    public void InputAbcAndLeft()
    {
        _textbox.Keyin("abc");
        _textbox.OnInput(new InputEvent
        {
            Key = ConsoleKey.LeftArrow
        });
        Assert.That(_textbox.EditIndex, Is.EqualTo(2));
        Assert.That(_textbox.Value, Is.EqualTo("abc"));
        
        _textbox.Keyin("1");
        Assert.That(_textbox.EditIndex, Is.EqualTo(3));
        Assert.That(_textbox.Value, Is.EqualTo("ab1c"));
        
        _textbox.Keyin("2");
        Assert.That(_textbox.EditIndex, Is.EqualTo(4));
        Assert.That(_textbox.Value, Is.EqualTo("ab12c"));
        
        _textbox.Keyin("3");
        Assert.That(_textbox.EditIndex, Is.EqualTo(5));
        Assert.That(_textbox.Value, Is.EqualTo("ab123c"));
    }
    
    [Test]
    public void Input11Chars()
    {
        _textbox.Keyin("12345678901");
        Assert.That(_textbox.EditIndex, Is.EqualTo(11));
        Assert.That(_textbox.Value, Is.EqualTo("12345678901"));

        Assert.That(_textbox[new Position {X = 10, Y = 11}],
            Is.EqualTo(Character.Empty));
    }
    
}

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
    }
}    