namespace VimSharpLib;
using System;
using System.Collections.Generic;

public interface IKeyPattern
{
    bool IsMatch(List<ConsoleKeyInfo> keyBuffer);
} 