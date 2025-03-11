namespace VimSharpLib;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;

/// <summary>
/// ConsoleKey的擴展方法
/// </summary>
public static class ConsoleKeyExtension
{
    /// <summary>
    /// 將ConsoleKey轉換為可讀的字符串表示
    /// </summary>
    public static string ToDisplayString(this IEnumerable<ConsoleKey> keyBuffer)
    {
        if (keyBuffer == null || !keyBuffer.Any())
            return string.Empty;
            
        var sb = new StringBuilder();
        
        foreach (var key in keyBuffer)
        {
            // 將 ConsoleKey 轉換為字符串表示
            string keyString = key.ToString();
            
            // 對於特殊按鍵，可以使用特定的表示方式
            // 例如，將方向鍵轉換為特定的字符
            switch (key)
            {
                case ConsoleKey.LeftArrow:
                    sb.Append("←");
                    break;
                case ConsoleKey.RightArrow:
                    sb.Append("→");
                    break;
                case ConsoleKey.UpArrow:
                    sb.Append("↑");
                    break;
                case ConsoleKey.DownArrow:
                    sb.Append("↓");
                    break;
                case ConsoleKey.Enter:
                    sb.Append("↵");
                    break;
                case ConsoleKey.Escape:
                    sb.Append("Esc");
                    break;
                case ConsoleKey.Tab:
                    sb.Append("Tab");
                    break;
                case ConsoleKey.Backspace:
                    sb.Append("⌫");
                    break;
                case ConsoleKey.Delete:
                    sb.Append("Del");
                    break;
                case ConsoleKey.Home:
                    sb.Append("Home");
                    break;
                case ConsoleKey.End:
                    sb.Append("End");
                    break;
                case ConsoleKey.PageUp:
                    sb.Append("PgUp");
                    break;
                case ConsoleKey.PageDown:
                    sb.Append("PgDn");
                    break;
                case ConsoleKey.Insert:
                    sb.Append("Ins");
                    break;
                // 數字鍵
                case ConsoleKey.D0:
                    sb.Append("0");
                    break;
                case ConsoleKey.D1:
                    sb.Append("1");
                    break;
                case ConsoleKey.D2:
                    sb.Append("2");
                    break;
                case ConsoleKey.D3:
                    sb.Append("3");
                    break;
                case ConsoleKey.D4:
                    sb.Append("4");
                    break;
                case ConsoleKey.D5:
                    sb.Append("5");
                    break;
                case ConsoleKey.D6:
                    sb.Append("6");
                    break;
                case ConsoleKey.D7:
                    sb.Append("7");
                    break;
                case ConsoleKey.D8:
                    sb.Append("8");
                    break;
                case ConsoleKey.D9:
                    sb.Append("9");
                    break;
                // 字母鍵
                default:
                    if (key >= ConsoleKey.A && key <= ConsoleKey.Z)
                    {
                        sb.Append((char)('a' + (key - ConsoleKey.A)));
                    }
                    else
                    {
                        // 其他未識別的按鍵，使用其名稱
                        sb.Append(keyString);
                    }
                    break;
            }
        }
        
        return sb.ToString();
    }
    
    /// <summary>
    /// 將ConsoleKey轉換為字符，用於正則表達式匹配
    /// </summary>
    public static char ToChar(this ConsoleKey key)
    {
        // 處理特殊字符
        switch (key)
        {
            case ConsoleKey.D0: return '0';
            case ConsoleKey.D1: return '1';
            case ConsoleKey.D2: return '2';
            case ConsoleKey.D3: return '3';
            case ConsoleKey.D4: return '4';
            case ConsoleKey.D5: return '5';
            case ConsoleKey.D6: return '6';
            case ConsoleKey.D7: return '7';
            case ConsoleKey.D8: return '8';
            case ConsoleKey.D9: return '9';
            case ConsoleKey.J: return 'J';
            // 處理字母
            default:
                if (key >= ConsoleKey.A && key <= ConsoleKey.Z)
                {
                    return (char)('a' + (key - ConsoleKey.A));
                }
                return '\0';
        }
    }
} 