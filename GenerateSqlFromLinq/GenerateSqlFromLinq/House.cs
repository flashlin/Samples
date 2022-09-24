using System.ComponentModel.DataAnnotations.Schema;

[Table("House")]
public class House
{
	public int Id { get; set; }
	public string Address { get; set; }
	public DateTime BuyTime { get; set; }
	public int CustomerId { get; set; }
}