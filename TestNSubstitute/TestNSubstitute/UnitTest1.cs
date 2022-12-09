using NSubstitute;

namespace TestNSubstitute;

public class Tests
{
    private IMyRepo _myRepo = null!;
    private MyService _myService = null!;

    [SetUp]
    public void Setup()
    {
        _myRepo = Substitute.For<IMyRepo>();
        _myService = new MyService(_myRepo);
    }

    [Test]
    public void check_received_args()
    {
        var result = _myService.SayHello("flash");
        
        _myRepo.Received().Insert(Arg.Do<List<string>>(x => x.SequenceEqual(new List<string>()
        {
            "flash"
        })));
    }
}