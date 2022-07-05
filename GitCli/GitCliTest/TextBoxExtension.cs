using GitCli.Models.ConsoleMixedReality;

namespace GitCliTest;

public static class ConsoleTextBoxExtension
{
    private static Dictionary<char, InputEvent> _keyDict;

    static ConsoleTextBoxExtension()
    {
        _keyDict = QueryKeys()
            .ToDictionary(x => x.KeyChar, x => x);
    }

    public static void Keyin(this IConsoleElement element, string text)
    {
        foreach (var ch in text)
        {
            var keyEvent = _keyDict[ch];
            element.OnInput(keyEvent);
        }
    }

    public static void ShiftKey(this IConsoleElement element, ConsoleKey key, bool hasShift = true)
    {
        element.OnInput(new InputEvent
        {
            Key = key,
            HasShift = hasShift
        });
    }

    private static IEnumerable<InputEvent> QueryKeys()
    {
        var keys = "abcdefghijklmnopqrstuvwxyz";
        foreach (var ch in keys)
        {
            var key = (ConsoleKey) Enum.Parse(typeof(ConsoleKey), $"{ch}".ToUpper());
            yield return new InputEvent
            {
                Key = key,
                KeyChar = ch
            };
        }

        var nums = "0123456789";
        foreach (var num in nums)
        {
            var key = (ConsoleKey) Enum.Parse(typeof(ConsoleKey), $"D{num}");
            yield return new InputEvent
            {
                Key = key,
                KeyChar = num
            };
        }
    }
}