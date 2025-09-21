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
    public async Task Test1()
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

    public void Dispose()
    {
        _demoServerApp?.Dispose();
    }
}