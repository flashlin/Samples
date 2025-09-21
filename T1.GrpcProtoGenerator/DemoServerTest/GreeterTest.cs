using DemoServer.Services;

namespace DemoServerTest;

public class Tests : IDisposable
{
    private DemoServerApp? _demoServerApp;

    [SetUp]
    public void Setup()
    {
        _demoServerApp = new DemoServerApp();
        _demoServerApp.Initialize();
    }

    [Test]
    public async Task TestSayHello()
    {
        // Arrange
        var request = new HelloRequestGrpcDto
        {
            Name = "Test User"
        };

        // Act
        var response = await _demoServerApp!.Client.SayHelloAsync(request);

        // Assert
        Assert.That(response, Is.Not.Null);
        Assert.That(response.Message, Is.EqualTo("Hello Test User!"));
    }

    [Test]
    public async Task TestSayTime()
    {
        // Arrange
        var startTime = DateTime.UtcNow;
        var request = new GetTimeRequestGrpcDto
        {
            Id = 123,
            StartTime = startTime
        };

        // Act
        var response = await _demoServerApp!.Client.SayTimeAsync(request);

        // Assert
        Assert.That(response, Is.Not.Null);
        Assert.That(response.ReplyTime, Is.EqualTo(startTime.AddSeconds(1)));
    }

    public void Dispose()
    {
        _demoServerApp?.Dispose();
    }
}