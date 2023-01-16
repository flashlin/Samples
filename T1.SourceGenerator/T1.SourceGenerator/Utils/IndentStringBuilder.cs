using System.Text;

namespace T1.SourceGenerator.Utils;

public class IndentStringBuilder
{
    private readonly object _lockBase = new object();

    private readonly StringBuilder _sb = new StringBuilder();

    public bool Written { get; set; }

    public int Indent { get; set; }

    public string IndentChars { get; set; } = "\t";


    public void WriteLine(string str, params object[] args)
    {
        Write(str, args);
        lock (_lockBase)
        {
            _sb.AppendLine();
            Written = false;
        }
    }

    public void WriteLine()
    {
        WriteLine("");
    }

    public void Write(string str, params object[] args)
    {
        lock (_lockBase)
        {
            if (!Written)
            {
                for (int i = 0; i < Indent; i++)
                {
                    _sb.Append(IndentChars);
                }

                Written = true;
            }
        }

        string value = ((args == null || args.Length == 0) ? str : string.Format(str, args));
        _sb.Append(value);
    }

    public void WriteText(string text)
    {
        StringReader stringReader = new StringReader(text);
        while (true)
        {
            string text2 = stringReader.ReadLine();
            if (text2 == null)
            {
                break;
            }

            WriteLine(text2);
        }
    }


    public override string ToString()
    {
        return _sb.ToString();
    }
}