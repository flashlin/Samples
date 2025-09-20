using System;
using System.Text;

namespace T1.GrpcProtoGenerator.Common;

/// <summary>
/// Simple IndentStringBuilder for Source Generator use (no third-party dependencies)
/// </summary>
internal class IndentStringBuilder
{
    private readonly StringBuilder _sb = new StringBuilder();
    private int _indent = 0;
    private const string IndentString = "    "; // 4 spaces per indent level

    public int Indent
    {
        get => _indent;
        set => _indent = Math.Max(0, value);
    }

    public void WriteLine()
    {
        _sb.AppendLine();
    }

    public void WriteLine(string value)
    {
        if (!string.IsNullOrEmpty(value))
        {
            for (int i = 0; i < _indent; i++)
            {
                _sb.Append(IndentString);
            }
        }
        _sb.AppendLine(value);
    }

    public override string ToString()
    {
        return _sb.ToString();
    }
}