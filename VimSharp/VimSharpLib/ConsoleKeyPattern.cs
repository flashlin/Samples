namespace VimSharpLib;
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Linq;

public class AnyKeyPattern : IKeyPattern
{
    public bool IsMatch(List<ConsoleKey> keyBuffer)
    {
        return true;
    }
}

public class ConsoleKeyPattern : IKeyPattern
{
    private readonly IEnumerable<ConsoleKey> _keys;

    public ConsoleKeyPattern(IEnumerable<ConsoleKey> keys)
    {
        _keys = keys ?? throw new ArgumentNullException(nameof(keys));
    }
    
    public ConsoleKeyPattern(ConsoleKey key)
        : this([key])
    {
    }

    public bool IsMatch(List<ConsoleKey> keyBuffer)
    {
        if (keyBuffer == null || keyBuffer.Count == 0)
            return false;

        // 完全比對：檢查按鍵緩衝區是否與指定的按鍵序列完全匹配
        // 首先檢查長度是否相同
        var keysList = _keys.ToList();
        if (keyBuffer.Count != keysList.Count)
            return false;

        // 然後檢查每個位置的按鍵是否相同
        for (int i = 0; i < keyBuffer.Count; i++)
        {
            if (keyBuffer[i] != keysList[i])
                return false;
        }

        return true;
    }
} 