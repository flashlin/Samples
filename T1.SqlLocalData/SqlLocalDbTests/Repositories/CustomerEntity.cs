using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace SqlLocalDbTests.Repositories;

[Table("Customer")]
public class CustomerEntity
{
	[Key]
	public int Id { get; set; }
	
	[StringLength(50)]
	public string? Name { get; set; }
}
