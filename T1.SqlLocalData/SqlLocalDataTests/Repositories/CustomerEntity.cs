using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace SqlLocalDataTests.Repositories;

[Table("Customer")]
public class CustomerEntity
{
	[Key]
	public int Id { get; set; }
	[StringLength(50)]
	public string Name { get; set; }
	public DateTime Birth { get; set; }
}
