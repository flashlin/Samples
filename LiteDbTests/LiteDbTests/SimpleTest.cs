using FluentAssertions;
using LiteDB;

namespace LiteDbTests
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
            if (File.Exists("MyData.db"))
            {
                File.Delete("MyData.db");
            }
        }

        [Test]
        public void TestAdd()
        {
            var customer = new Customer
            {
                Name = "John",
                Phones = new string[] { "8000-0000", "9000-0000" },
                IsActive = true
            };
            var actual = AddCustomer(customer);
            actual.Should().BeEquivalentTo(customer);
        }

        private static Customer AddCustomer(Customer customer)
        {
            var repo = new UserRepo();
            var actual = repo.Add(customer);
            return actual;
        }


        [Test]
        public void TestUpdate()
        {
            AddCustomer(new Customer()
            {
                Name = "John",
                Phones = new string[] { "8000-0000", "9000-0000" },
                IsActive = true
            });
            
            var customer = new Customer
            {
                Id = 1,
                Name = "Flash",
                Phones = new string[] { "8000-0001" },
                IsActive = true
            };
            var repo = new UserRepo();
            repo.Update(customer);
            var actual = repo.QueryByName("Flash");
            actual.Should().BeEquivalentTo(customer);
        }

        [Test]
        public void TestDelete()
        {
            AddCustomer(new Customer()
            {
                Name = "Flash",
                Phones = new string[] { "8000-0000", "9000-0000" },
                IsActive = true
            });

            var repo = new UserRepo();
            repo.Delete(1);

            var actual = repo.QueryByName("Flash");
            actual.Should().BeNull();
        }
    }

    public class Customer
    {
        public int Id { get; set; }
        public string Name { get; init; } = string.Empty;
        public string[] Phones { get; init; } = Array.Empty<string>();
        public bool IsActive { get; init; }
    }

    public class UserRepo
    {
        public Customer Add(Customer customer)
        {
            using var db = new LiteDatabase(@"MyData.db");
            var col = db.GetCollection<Customer>("customers");
            col.Insert(customer);
            return customer;
        }

        public void Update(Customer customer)
        {
            using var db = new LiteDatabase(@"MyData.db");
            var col = db.GetCollection<Customer>("customers");
            col.FindById(customer.Id);
            col.Update(customer);
        }

        public Customer QueryByName(string name)
        {
            using var db = new LiteDatabase(@"MyData.db");
            var col = db.GetCollection<Customer>("customers");
            col.EnsureIndex(x => x.Name);

            var item = col.Query()
                .Where(x => x.Name == name)
                .FirstOrDefault();
            return item;
        }

        public void Delete(int id)
        {
            using var db = new LiteDatabase(@"MyData.db");
            var col = db.GetCollection<Customer>("customers");
            col.Delete(id);
        }
    }
}