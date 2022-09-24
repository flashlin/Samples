using System.ComponentModel.DataAnnotations.Schema;

[Table("Customer")]
public class Customer
{
	public int Id { get; set; }
	public string Name { get; set; }
}