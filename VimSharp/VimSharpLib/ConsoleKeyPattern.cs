namespace VimSharpLib;
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Linq;

public class ConsoleKeyPattern : IKeyPattern
{
    private readonly ConsoleKey _key;

    public ConsoleKeyPattern(ConsoleKey key)
    {
        _key = key;
    }

    public bool IsMatch(List<ConsoleKey> keyBuffer)
    {
        if (keyBuffer == null || keyBuffer.Count == 0)
            return false;

        // 檢查按鍵緩衝區中是否包含指定的按鍵
        return keyBuffer.Contains(_key);
    }
} 