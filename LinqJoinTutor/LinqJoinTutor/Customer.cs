using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace LinqJoinTutor;

[Table("Customer")]
public class Customer
{
	[Key]
	[DatabaseGenerated(DatabaseGeneratedOption.Identity)]
	public int Id { get; set; }
	public string Name { get; set; }
	public DateTime Birth { get; set; }
	public int Price { get; set; }
}