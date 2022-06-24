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



		public void Dotnet6_Array()
		{
			var name1 = new[] { "Flash" };
			var name2 = new[] { "Jack" };
			var name3 = new[] { "Mary" };

			var names = name1.Concat(name2).Concat(name3);

			var count = names.Count();
			var newNames = names.OrderBy(x => x);
		}







	}
}
