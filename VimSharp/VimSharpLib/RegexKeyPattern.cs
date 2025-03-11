namespace VimSharpLib;
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Linq;

public class RegexKeyPattern : IKeyPattern
{
    private readonly string _pattern;
    private readonly Dictionary<string, ConsoleKey> _keyMapping;

    public RegexKeyPattern(string pattern)
    {
        _pattern = pattern;
        _keyMapping = InitializeKeyMapping();
    }

    private Dictionary<string, ConsoleKey> InitializeKeyMapping()
    {
        var mapping = new Dictionary<string, ConsoleKey>();
        
        // 基本按鍵映射
        mapping.Add("I", ConsoleKey.I);
        mapping.Add("A", ConsoleKey.A);
        mapping.Add("Q", ConsoleKey.Q);
        mapping.Add("Left", ConsoleKey.LeftArrow);
        mapping.Add("Right", ConsoleKey.RightArrow);
        mapping.Add("Up", ConsoleKey.UpArrow);
        mapping.Add("Down", ConsoleKey.DownArrow);
        mapping.Add("Enter", ConsoleKey.Enter);
        
        return mapping;
    }

    public bool IsMatch(List<ConsoleKey> keyBuffer)
    {
        if (keyBuffer == null || keyBuffer.Count == 0)
            return false;

        // 將按鍵緩衝區轉換為字符串表示
        string keySequence = string.Join("", keyBuffer.Select(k => GetKeyString(k)));
        
        // 使用正則表達式匹配
        return Regex.IsMatch(keySequence, _pattern);
    }

    private string GetKeyString(ConsoleKey key)
    {
        // 反向查找按鍵映射
        foreach (var pair in _keyMapping)
        {
            if (pair.Value == key)
                return pair.Key;
        }
        
        // 如果找不到映射，返回按鍵的字符串表示
        return key.ToString();
    }
} 