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

[Table("House")]
public class House
{
	[Key]
	[DatabaseGenerated(DatabaseGeneratedOption.Identity)]
	public int Id { get; set; }
	public int CustomerId { get; set; }
	public string Address { get; set; }
	public DateTime BuyTime { get; set; }
}
