namespace VimSharpLib;
using System;
using System.Collections.Generic;
using System.Linq;

public class CharKeyPattern : IKeyPattern
{
    private readonly char _char;

    public CharKeyPattern(char c)
    {
        _char = c;
    }

    public bool IsMatch(List<ConsoleKeyInfo> keyBuffer)
    {
        if (keyBuffer.Count == 0)
            return false;

        // 檢查按鍵緩衝區的最後一個按鍵是否對應於指定的字符
        var lastKeyChar = keyBuffer.Last().KeyChar;
        
        // 處理特殊符號
        return _char == lastKeyChar;
    }
} 