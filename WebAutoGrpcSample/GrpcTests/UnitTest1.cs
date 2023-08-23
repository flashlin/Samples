using FluentAssertions;
using Shared.Contracts;

namespace GrpcTests;

public class Tests
{
    [SetUp]
    public void Setup()
    {
    }

    [Test]
    public void Test1()
    {
        // var obj1 = new User1
        // {
        //     Id = 1,
        //     Name = "Flash"
        // };
        // var sut = new BsonSerializer();
        // var bytes1 = sut.ToBytes(obj1);
        // var obj2 = sut.FromBytes<User2>(bytes1);
        // obj2.Should().Be(new User2
        // {
        //     Id = 1,
        //     Name = "Flash",
        //     Price = 0
        // });
    }

    public class User1
    {
        public int Id { get; set; }
        public string Name { get; set; }
    }
    
    public class User2
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public decimal Price { get; set; }
    }
}