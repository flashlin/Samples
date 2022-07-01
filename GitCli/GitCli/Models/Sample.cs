using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace GitCli.Models
{
	public class Example
	{
		public class Person
		{
			public string? Name { get; set; }
			public int Age { get; set; }
		}

		public class Student : Person
		{
			public int Grade { get; set; }
		}
		
		public void Sample()
		{
			//Dotnet6_Index();
			Dotnet6_Take();
		}

		public void Dotnet6_Serialize()
		{
			var person = new Person { Name = "Flash", Age = 60 };
			var personAsJson = JsonSerializer.Serialize(person);

			Person student = new Student { Name = "Flash", Age = 60, Grade = 12 };
			var studentAsJson = JsonSerializer.Serialize(student, student.GetType());
		}

		public void Dotnet6_ChunkArray()
		{
			var names = new[] {
				"Flash", "Jack", "Jason", "April", "Frank", "Mary", "Mar", "Stephen", "Curry"
			};
			var chunk = names.Chunk(3);
		}



		public void Dotnet6_TryGetNonEnumeratedCount()
		{
			var name1 = new[] { "Flash" };
			var name2 = new[] { "Jack" };
			var name3 = new[] { "Mary" };

			var names = name1.Concat(name2).Concat(name3);

			var count = names.Count();
			var newNames = names.OrderBy(x => x);

			names.TryGetNonEnumeratedCount(out var count2);
		}



		public void Dotnet6_Zip()
		{
			var names = new[] { "Flash","Jack", "Mary" };
			var args = new[] {1, 2, 3};
			var combines = names.Zip(args);
		}


		public void Dotnet6_Min()
		{
			var args = new[] {1, 2, 3};
			var min1 = args.OrderBy(x => x).First();
			var min = args.Min(x => x);
			var max = args.Max(x => x);
		}

		public void Dotnet6_Index()
		{
			var args = new[] {1, 2, 3, 4, 5};
			var a = args.ElementAt(^2);
		}
		
		public void Dotnet6_Take()
		{
			var args = new[] {1, 2, 3, 4, 5};
			//不包含 4
			var a = args.Take(2..4);
			var b = args.Take(^3..);
		}
	}
}
