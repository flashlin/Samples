namespace RecursiveDescentParserDemo.ParseEx1;

public interface IEnumerableStream<out T>
{
    T Current { get; }
    void Move(int position);
    int GetPosition();
    bool IsEof();
    bool MoveNext();
}