namespace RecursiveDescentParserDemo.ParseEx1;

public interface IMatcher<in T>
{
    bool Match(T input);
}