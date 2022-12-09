namespace TestNSubstitute;

public class MyService
{
    private IMyRepo _myRepo;

    public MyService(IMyRepo myRepo)
    {
        _myRepo = myRepo;
    }
    
    public string SayHello(string name)
    {
        _myRepo.Insert(new List<string>()
        {
            name
        });
        return $"Hi, {name}.";
    }
}