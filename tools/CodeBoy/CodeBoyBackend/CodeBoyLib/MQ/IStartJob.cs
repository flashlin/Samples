namespace CodeBoyLib.MQ;

public interface IStartJob
{
    string JobId { get; }
    Task Execute();
}
